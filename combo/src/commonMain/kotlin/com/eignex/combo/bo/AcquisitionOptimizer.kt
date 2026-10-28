package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.FeatureEncoder
import com.eignex.combo.decisions.conditionedProblem
import com.eignex.combo.decisions.unframed
import com.eignex.combo.shared.SolverCoordinateLayout
import com.eignex.combo.shared.WitnessRequest
import com.eignex.combo.shared.WitnessSearch
import com.eignex.combo.shared.WitnessTermination
import com.eignex.combo.shared.WitnessValidator
import com.eignex.combo.shared.assignmentKey
import com.eignex.combo.shared.interpolateWitness
import com.eignex.combo.util.checkedInt
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

/** Bounded multi-start neighborhood policy in normalized decision geometry. */
data class LocalSearchPolicy(
    /** Number of independently refined initial points. */
    val starts: Int = 4,
    /** Consecutive non-improving steps allowed before ending a walk. */
    val plateauSteps: Int = 5,
    /** Initial normalized proposal radius. */
    val initialRadius: Double = 0.25,
    /** Minimum adaptive radius. */
    val minRadius: Double = 0.005,
    /** Maximum adaptive radius after consecutive improvements. */
    val maxRadius: Double = 0.5,
    /** False with one start retains the fixed shrinking-radius search. */
    val adaptive: Boolean = true,
) {
    init {
        require(starts > 0 && plateauSteps > 0)
        require(minRadius > 0.0 && minRadius <= initialRadius && initialRadius <= maxRadius && maxRadius <= 1.0)
    }
}

private data class ScoredPoint(val sample: Sample, val score: Double)

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
 * @property localPolicy Multi-start, plateau, and adaptive-radius limits.
 */
class LocalAndRandomSearch(
    val projection: FeatureEncoder<com.eignex.koblas.Vector>,
    val witnesses: WitnessSearch,
    val candidateCount: Int = 64,
    val maxAttempts: Int = 128,
    val localSteps: Int = 8,
    val neighbors: Int = 12,
    val tieBreak: AcquisitionTieBreak = AcquisitionTieBreak.Random,
    val localPolicy: LocalSearchPolicy = LocalSearchPolicy(),
) : AcquisitionOptimizer {
    private val space = projection.space
    private val validator = WitnessValidator(space)
    private val neighborhood = FeasibleNeighborhood(space)
    init {
        require(candidateCount > 0 && maxAttempts >= 0 && localSteps >= 0 && neighbors > 0)
    }

    override fun maximize(acquisition: Acquisition, request: AcquisitionRequest, random: Random): CandidateSelection {
        val assumptions = space.discreteAssumptionsFor(request.context)
        val excluded = request.excluded.map { it.assignmentKey() }.toSet()
        val anchors = request.anchors.filter { space.matches(it, request.context) && validator.valid(it) }
        val generated = witnesses.witnesses(
            WitnessRequest(
                candidateCount,
                assumptions,
                anchors = anchors.take(candidateCount),
                exclude = request.excluded,
                maxAttempts = maxAttempts,
                context = request.context,
            ),
            random,
        )
        val explored = linkedMapOf<com.eignex.combo.shared.AssignmentKey, Sample>()
        val scored = linkedMapOf<com.eignex.combo.shared.AssignmentKey, ScoredPoint>()
        var evaluations = 0
        var best: Sample? = null
        var bestScore = Double.NEGATIVE_INFINITY
        var bestTies = 0
        var guide: Sample? = null
        var guideScore = Double.NEGATIVE_INFINITY
        var guideTies = 0
        fun evaluate(points: List<Sample>) {
            val valid = points.distinctBy { it.assignmentKey() }.filter {
                it.assignmentKey() !in explored && space.matches(it, request.context) && validator.valid(it)
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
                scored[point.assignmentKey()] = ScoredPoint(point, scores[i])
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
        if (!localPolicy.adaptive && localPolicy.starts == 1) {
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
                        request.context,
                    ),
                )
            }
        } else if (localSteps > 0) {
            val initial = if (tieBreak == AcquisitionTieBreak.Random) {
                scored.values.shuffled(
                    random,
                )
            } else {
                scored.values.toList()
            }
            val starters = initial.sortedByDescending { it.score }.take(localPolicy.starts)
            for (start in starters) {
                var current = start
                var radius = localPolicy.initialRadius
                var plateau = 0
                var improvements = 0
                val visited = mutableSetOf(current.sample.assignmentKey())
                for (step in 0 until localSteps) {
                    val proposals = neighborhood.around(
                        current.sample,
                        assumptions,
                        random,
                        radius,
                        neighbors,
                        explored.values.toList(),
                        request.context,
                    )
                    evaluate(proposals)
                    val available = proposals.mapNotNull { scored[it.assignmentKey()] }
                        .filter { it.sample.assignmentKey() !in visited }
                    val ordered = if (tieBreak == AcquisitionTieBreak.Random) available.shuffled(random) else available
                    val next = ordered.maxByOrNull { it.score }
                    if (next != null && next.score > current.score) {
                        current = next
                        visited += next.sample.assignmentKey()
                        plateau = 0
                        improvements++
                        if (improvements >= 3) {
                            radius = minOf(localPolicy.maxRadius, radius * 2.0)
                            improvements = 0
                        }
                    } else {
                        plateau++
                        improvements = 0
                        if (next != null && next.score == current.score) {
                            current = next
                            visited += next.sample.assignmentKey()
                        }
                        if (plateau % 2 == 0) radius = maxOf(localPolicy.minRadius, radius * 0.5)
                        if (plateau >= localPolicy.plateauSteps) break
                    }
                }
            }
        }
        val termination = if (best == null && generated.termination == WitnessTermination.CountReached) {
            WitnessTermination.BudgetExhausted
        } else {
            generated.termination
        }
        return CandidateSelection(best?.let(::BanditSample), termination, evaluations)
    }
}

internal class FeasibleNeighborhood(private val space: CompiledDecisionSpace) {
    private val validator = WitnessValidator(space)
    private val solver by lazy { BacktrackSolver(space.coordinates.problem.bake()) }
    private val coordinates = SolverCoordinateLayout(space)
    private val realIds = coordinates.realHandles
    private val names = (space.coordinates.boolVarIdByName.keys + space.numericDomains.keys).distinct()

    fun around(
        anchor: Sample,
        assumptions: Assumptions,
        random: Random,
        radius: Double,
        count: Int,
        seeds: List<Sample>,
        context: Context = Context.Empty,
    ): List<Sample> {
        if (space.coordinates.schema == null || names.isEmpty()) return emptyList()
        val scopedSolver = if (!context.hasRealValues) {
            solver
        } else {
            BacktrackSolver(space.conditionedProblem(context).bake())
        }
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
                return@List interpolateWitness(anchor, peer, fraction)?.takeIf {
                    space.matches(it, assumptions) &&
                        (context === Context.Empty || space.matches(it, context)) && validator.valid(it)
                }
            }
            val name = names[random.nextInt(names.size)]
            if (space.contextFloats.any { it.name == name }) return@List null
            val boolId = space.coordinates.boolVarIdByName[name]
            val intId = space.coordinates.intVarIdByName[name]
            val realId = realIds.entries.firstOrNull { it.value.name == name }?.key
            if (boolId != null && boolId !in assumptions.bools) {
                scopedSolver.sample(
                    BacktrackParams(
                        maxDecisions = 256L,
                        randomSeed = random.nextLong(),
                        assumptions = assumptions.mergedWith(
                            Assumptions(bools = mapOf(boolId to !anchor.boolValue(boolId))),
                        ),
                    ),
                ).assignment?.unframed()
            } else if (intId != null && intId !in assumptions.ints) {
                val domain = space.numericDomains[name] ?: return@List null
                val lo = domain.feature(domain.min)
                val hi = domain.feature(domain.max)
                val spec = coordinates.floatSpecs[intId]
                val sourceValue = spec?.realValue(
                    checkedInt(anchor.intValue(intId)),
                ) ?: anchor.intValue(intId).toDouble()
                val unit = if (lo == hi) 0.0 else (domain.feature(sourceValue) - lo) / (hi - lo)
                val proposed = domain.fromUnit((unit + random.nextDouble(-radius, radius)).coerceIn(0.0, 1.0))
                val value = if (spec == null) {
                    proposed.roundToLong()
                } else if (spec.scale == 0.0) {
                    0L
                } else {
                    ((proposed - spec.min) / spec.scale).roundToLong()
                }
                scopedSolver.sample(
                    BacktrackParams(
                        maxDecisions = 256L,
                        randomSeed = random.nextLong(),
                        assumptions = assumptions.mergedWith(Assumptions(ints = mapOf(intId to value))),
                    ),
                ).assignment?.unframed()
            } else if (realId != null) {
                val domain = space.numericDomains[name] ?: return@List null
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
                point.takeIf {
                    space.matches(it, assumptions) &&
                        (context === Context.Empty || space.matches(it, context)) && validator.valid(it)
                }
            } else {
                null
            }
        }.filterNotNull()
    }
}
