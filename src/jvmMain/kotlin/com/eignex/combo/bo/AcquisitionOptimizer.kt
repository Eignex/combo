package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.FeatureEncoder
import com.eignex.combo.shared.SolverCoordinateLayout
import com.eignex.combo.shared.WitnessRequest
import com.eignex.combo.shared.WitnessSearch
import com.eignex.combo.shared.WitnessTermination
import com.eignex.combo.shared.WitnessValidator
import com.eignex.combo.shared.assignmentKey
import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.backtrack.BacktrackSolver
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.propagation.bake
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.Sample
import com.eignex.koblas.DenseVector
import kotlin.math.roundToLong
import kotlin.random.Random

/** One acquisition-search outcome, preserving budget status and work diagnostics. */
class CandidateSelection(
    /** Highest-scoring feasible point found, or null on a complete miss. */
    val sample: BanditSample?,
    /** Witness-generation termination; a returned incumbent remains valid on budget exhaustion. */
    val termination: WitnessTermination,
    /** Number of acquisition evaluations, including local-search neighbors. */
    val evaluations: Int,
)

/** Frozen hard context, exclusions, and history anchors for acquisition maximization. */
class AcquisitionRequest(
    /** Caller-supplied context pinned by Klause. */
    val context: Context,
    /** Pending assignments excluded from this search. */
    val excluded: List<Sample>,
    /** Observations useful as independently checked initial candidates. */
    val anchors: List<Sample>,
)

/** Acquisition-guided search over original feasible configurations. */
fun interface AcquisitionOptimizer {
    /** Maximize a fixed acquisition function; returned points must obey request context and exclusions. */
    fun maximize(acquisition: Acquisition, request: AcquisitionRequest, random: Random): CandidateSelection
}

/** How equal acquisition scores are ordered, including the flat regions of tree models. */
enum class AcquisitionTieBreak {
    /** Preserve candidate order; useful for reproducing previous search behavior. */
    First,

    /** Seeded reservoir sampling among distinct tied points, allowing plateau exploration. */
    Random,
}

/**
 * Random initialization followed by score-guided feasible neighborhood search.
 * Every accepted neighbor is checked against the original Klause problem. Integer and Boolean
 * moves may invoke bounded solver repair; real moves are independently checked before scoring.
 *
 * @property projection Model feature encoder; no constraints are rewritten into model coordinates.
 * @property witnesses Batch feasible initialization provider.
 * @property candidateCount Requested initial candidate count.
 * @property maxAttempts Initialization proposal allowance.
 * @property localSteps Number of successive neighborhood searches.
 * @property neighbors Number of attempted neighbors per local step.
 * @property tieBreak Ordering of exactly equal acquisition scores.
 */
class LocalAndRandomSearch(
    val projection: FeatureEncoder<com.eignex.koblas.Vector>,
    val witnesses: WitnessSearch,
    val candidateCount: Int = 64,
    val maxAttempts: Int = 128,
    val localSteps: Int = 8,
    val neighbors: Int = 12,
    val tieBreak: AcquisitionTieBreak = AcquisitionTieBreak.Random,
) : AcquisitionOptimizer {
    private val space = projection.space
    private val validator = WitnessValidator(space)
    private val neighborhood = FeasibleNeighborhood(space)
    init {
        require(candidateCount > 0 && maxAttempts >= 0 && localSteps >= 0 && neighbors > 0)
    }

    override fun maximize(acquisition: Acquisition, request: AcquisitionRequest, random: Random): CandidateSelection {
        val assumptions = space.assumptionsFor(request.context)
        val excluded = request.excluded.map { it.assignmentKey() }.toSet()
        val anchors = request.anchors.filter { space.matches(it, assumptions) && validator.valid(it) }
        val generated = witnesses.witnesses(
            WitnessRequest(
                candidateCount,
                assumptions,
                anchors = anchors.take(candidateCount),
                exclude = request.excluded,
                maxAttempts = maxAttempts,
            ),
            random,
        )
        val explored = linkedMapOf<com.eignex.combo.shared.AssignmentKey, Sample>()
        var evaluations = 0
        var best: Sample? = null
        var bestScore = Double.NEGATIVE_INFINITY
        var bestTies = 0
        var guide: Sample? = null
        var guideScore = Double.NEGATIVE_INFINITY
        var guideTies = 0
        fun evaluate(points: List<Sample>) {
            val valid = points.distinctBy { it.assignmentKey() }.filter {
                it.assignmentKey() !in explored && space.matches(it, assumptions) && validator.valid(it)
            }
            if (valid.isEmpty()) return
            for (point in valid) explored[point.assignmentKey()] = point
            val rows = valid.map {
                DenseVector.of(projection.encode(BanditSample(it), request.context).toDoubleArray())
            }
            val scores = acquisition.evaluate(featureRows(rows, projection.featureSize))
            require(scores.size == valid.size)
            evaluations += valid.size
            valid.forEachIndexed { i, point ->
                require(scores[i].isFinite()) { "acquisition returned a nonfinite score" }
                if (guide == null || scores[i] > guideScore) {
                    guide = point
                    guideScore = scores[i]
                    guideTies = 1
                } else if (scores[i] == guideScore && tieBreak == AcquisitionTieBreak.Random) {
                    guideTies++
                    if (random.nextInt(guideTies) == 0) guide = point
                }
                if (point.assignmentKey() in excluded) return@forEachIndexed
                if (best == null || scores[i] > bestScore) {
                    best = point
                    bestScore = scores[i]
                    bestTies = 1
                } else if (scores[i] == bestScore && tieBreak == AcquisitionTieBreak.Random) {
                    bestTies++
                    if (random.nextInt(bestTies) == 0) best = point
                }
            }
        }
        evaluate(anchors + generated.witnesses)
        repeat(localSteps) { step ->
            // An excluded maximizer may have no usable neighborhood; restart from other verified anchors.
            val anchor = if (best == null && step > 0 && anchors.isNotEmpty()) {
                anchors[(step - 1) % anchors.size]
            } else {
                guide ?: return@repeat
            }
            evaluate(
                neighborhood.around(
                    anchor,
                    assumptions,
                    random,
                    0.25 / (1.0 + step),
                    neighbors,
                    explored.values.toList(),
                ),
            )
        }
        val termination = if (best == null && generated.termination == WitnessTermination.CountReached) {
            WitnessTermination.BudgetExhausted
        } else {
            generated.termination
        }
        return CandidateSelection(best?.let(::BanditSample), termination, evaluations)
    }
}

private class FeasibleNeighborhood(private val space: CompiledDecisionSpace) {
    private val validator = WitnessValidator(space)
    private val solver by lazy { BacktrackSolver(space.compiled.problem.bake()) }
    private val realIds = SolverCoordinateLayout(space).realHandles
    private val names = (space.compiled.boolVarIdByName.keys + space.numericDomains.keys).distinct()

    fun around(
        anchor: Sample,
        assumptions: Assumptions,
        random: Random,
        radius: Double,
        count: Int,
        seeds: List<Sample>,
    ): List<Sample> {
        if (names.isEmpty()) return emptyList()
        val peers = if (realIds.isEmpty()) {
            emptyList()
        } else {
            seeds.filter {
                it.bools.contentEquals(
                    anchor.bools,
                ) && it.ints.contentEquals(anchor.ints) && it.assignmentKey() != anchor.assignmentKey()
            }
        }
        return List(count) { index ->
            if (peers.isNotEmpty() && index % 2 == 0) {
                val peer = peers[random.nextInt(peers.size)]
                val fraction = if (index == 0) 0.5 else random.nextDouble()
                return@List between(anchor, peer, fraction)?.takeIf {
                    space.matches(it, assumptions) && validator.valid(it)
                }
            }
            val name = names[random.nextInt(names.size)]
            val boolId = space.compiled.boolVarIdByName[name]
            val intId = space.compiled.intVarIdByName[name]
            val realId = realIds.entries.firstOrNull { it.value.name == name }?.key
            if (boolId != null && boolId !in assumptions.bools) {
                solver.sample(
                    BacktrackParams(
                        maxDecisions = 256L,
                        randomSeed = random.nextLong(),
                        assumptions = assumptions.mergedWith(
                            Assumptions(bools = mapOf(boolId to !anchor.boolValue(boolId))),
                        ),
                    ),
                ).assignment
            } else if (intId != null && intId !in assumptions.ints && name !in space.compiled.floatDecoders) {
                val domain = space.numericDomains.getValue(name)
                val lo = domain.feature(domain.min)
                val hi = domain.feature(domain.max)
                val unit = if (lo == hi) 0.0 else (domain.feature(anchor.intValue(intId).toDouble()) - lo) / (hi - lo)
                val value = domain.fromUnit(
                    (unit + random.nextDouble(-radius, radius)).coerceIn(0.0, 1.0),
                ).roundToLong()
                solver.sample(
                    BacktrackParams(
                        maxDecisions = 256L,
                        randomSeed = random.nextLong(),
                        assumptions = assumptions.mergedWith(Assumptions(ints = mapOf(intId to value))),
                    ),
                ).assignment
            } else if (realId != null) {
                val domain = space.numericDomains.getValue(name)
                val lo = domain.feature(domain.min)
                val hi = domain.feature(domain.max)
                val reals = anchor.reals
                val exact = (anchor.exactReals ?: reals.map { BigFraction.ofDouble(it)!! }).toMutableList()
                val unit = if (lo == hi) 0.0 else (domain.feature(reals[realId]) - lo) / (hi - lo)
                reals[realId] = domain.fromUnit((unit + random.nextDouble(-radius, radius)).coerceIn(0.0, 1.0))
                exact[realId] = BigFraction.ofDouble(reals[realId])!!
                val point = Sample(
                    anchor.bools,
                    anchor.ints,
                    reals,
                    exactReals = exact,
                )
                point.takeIf { space.matches(it, assumptions) && validator.valid(it) }
            } else {
                null
            }
        }.filterNotNull()
    }
    private fun between(left: Sample, right: Sample, fraction: Double): Sample? {
        val a = left.exactReals ?: left.reals.map { BigFraction.ofDouble(it) ?: return null }
        val b = right.exactReals ?: right.reals.map { BigFraction.ofDouble(it) ?: return null }
        val weight = BigFraction.ofDouble(fraction) ?: return null
        // Rational arithmetic preserves affine equalities; doubles must match the exact projections.
        val exact = a.indices.map { a[it] * (BigFraction.ONE - weight) + b[it] * weight }
        val coordinates = DoubleArray(exact.size) { index ->
            val projected = exact[index].toDouble()
            if (!projected.isFinite() || (projected == 0.0 && !exact[index].isZero)) return null
            projected
        }
        return Sample(left.bools, left.ints, coordinates, exactReals = exact)
    }
}
