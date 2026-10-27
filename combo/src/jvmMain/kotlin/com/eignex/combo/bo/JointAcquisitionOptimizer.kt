package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.FeatureEncoder
import com.eignex.combo.shared.WitnessRequest
import com.eignex.combo.shared.WitnessSearch
import com.eignex.combo.shared.WitnessTermination
import com.eignex.combo.shared.WitnessValidator
import com.eignex.combo.shared.assignmentKey
import com.eignex.koblas.Vector
import kotlin.random.Random

/** Feasible joint search result, with complete-set acquisition evaluation accounting. */
class JointCandidateSelection(
    /** Distinct proposed configurations; fewer than requested when the pool is exhausted. */
    val samples: List<BanditSample>,
    /** Count reached or the underlying witness-pool termination. */
    val termination: WitnessTermination,
    /** Number of whole candidate sets scored. */
    val evaluations: Int,
)

/** Search for an entire batch under one fixed joint acquisition. */
fun interface JointAcquisitionOptimizer {
    /** Maximize over distinct, feasible, unexcluded configurations in the requested context. */
    fun maximize(
        acquisition: JointAcquisition,
        request: AcquisitionRequest,
        count: Int,
        random: Random,
    ): JointCandidateSelection
}

/**
 * Bounded joint search over one feasible witness pool. Multiple random sets undergo coordinate exchange,
 * evaluating the entire set for each replacement. This is joint set optimization, not a sum of marginal EIs.
 * Search is restricted to the generated pool; it makes no global-optimum guarantee.
 *
 * @property projection Numeric model encoder.
 * @property witnesses Feasible candidate pool provider.
 * @property candidateCount Requested pool size.
 * @property starts Number of independently initialized sets.
 * @property passes Maximum coordinate-exchange sweeps per start.
 */
class JointPoolSearch(
    val projection: FeatureEncoder<Vector>,
    val witnesses: WitnessSearch,
    val candidateCount: Int = 32,
    val starts: Int = 4,
    val passes: Int = 2,
) : JointAcquisitionOptimizer {
    init {
        require(candidateCount > 0 && starts > 0 && passes >= 0)
    }
    override fun maximize(
        acquisition: JointAcquisition,
        request: AcquisitionRequest,
        count: Int,
        random: Random,
    ): JointCandidateSelection = JointSearch(
        projection,
        witnesses,
        candidateCount,
        starts,
        passes,
        0,
        1,
        LocalSearchPolicy(),
    ).maximize(acquisition, request, count, random)
}

/**
 * Multi-start whole-set search, followed by feasible local moves outside the initial witness pool.
 * One member moves at a time, with the entire batch re-scored. Reuses single-point search's checked
 * normalized neighborhoods and bounded Klause repair; returned configurations stay in original coordinates.
 *
 * @property projection Numeric model feature encoder.
 * @property witnesses Initial feasible candidate provider.
 * @property candidateCount Initial pool allowance, plus eligible history anchors.
 * @property passes Pool coordinate-exchange sweeps per start.
 * @property localSteps Maximum local sweeps per start.
 * @property neighbors Attempted feasible neighbors per member and sweep.
 * @property localPolicy Multi-start, plateau, and normalized-radius policy.
 */
class JointLocalAndRandomSearch(
    val projection: FeatureEncoder<Vector>,
    val witnesses: WitnessSearch,
    val candidateCount: Int = 32,
    val passes: Int = 2,
    val localSteps: Int = 8,
    val neighbors: Int = 12,
    val localPolicy: LocalSearchPolicy = LocalSearchPolicy(),
) : JointAcquisitionOptimizer {
    init {
        require(candidateCount > 0 && passes >= 0 && localSteps >= 0 && neighbors > 0)
    }
    override fun maximize(
        acquisition: JointAcquisition,
        request: AcquisitionRequest,
        count: Int,
        random: Random,
    ): JointCandidateSelection = JointSearch(
        projection,
        witnesses,
        candidateCount,
        localPolicy.starts,
        passes,
        localSteps,
        neighbors,
        localPolicy,
    )
        .maximize(acquisition, request, count, random)
}

private class JointSearch(
    private val projection: FeatureEncoder<Vector>,
    private val witnesses: WitnessSearch,
    private val candidateCount: Int,
    private val starts: Int,
    private val passes: Int,
    private val localSteps: Int,
    private val neighbors: Int,
    private val policy: LocalSearchPolicy,
) {
    fun maximize(
        acquisition: JointAcquisition,
        request: AcquisitionRequest,
        count: Int,
        random: Random,
    ): JointCandidateSelection {
        require(count > 0)
        val space = projection.space
        val assumptions = space.assumptionsFor(request.context)
        val validator = WitnessValidator(space)
        val excluded = request.excluded.map { it.assignmentKey() }.toSet()
        val anchors = request.anchors.filter { space.matches(it, assumptions) && validator.valid(it) }
        val generated = witnesses.witnesses(
            WitnessRequest(
                maxOf(candidateCount, count),
                assumptions,
                anchors = anchors,
                exclude = request.excluded,
                maxAttempts = (maxOf(candidateCount, count).toLong() * 2).coerceAtMost(Int.MAX_VALUE.toLong()).toInt(),
            ),
            random,
        )
        val pool = mutableListOf<BanditSample>()
        val indices = mutableMapOf<com.eignex.combo.shared.AssignmentKey, Int>()
        fun include(sample: com.eignex.klause.solver.Sample): Int? {
            val key = sample.assignmentKey()
            if (key in excluded || !space.matches(sample, assumptions) || !validator.valid(sample)) return null
            return indices.getOrPut(key) {
                pool.add(BanditSample(sample))
                pool.lastIndex
            }
        }
        (anchors + generated.witnesses).forEach { include(it) }
        val shortTermination = if (generated.termination ==
            WitnessTermination.CountReached
        ) {
            WitnessTermination.BudgetExhausted
        } else {
            generated.termination
        }
        if (pool.isEmpty()) return JointCandidateSelection(emptyList(), shortTermination, 0)
        val initial = pool.indices.toList()
        val size = minOf(count, pool.size)
        val cache = mutableMapOf<List<Int>, Double>()
        fun score(set: List<Int>): Double = cache.getOrPut(set.sorted()) {
            acquisition.evaluate(
                featureRows(set.sorted().map { projection.encode(pool[it], request.context) }, projection.featureSize),
            )
                .also { require(!it.isNaN() && it != Double.POSITIVE_INFINITY) { "invalid joint acquisition score" } }
        }
        val neighborhood = FeasibleNeighborhood(space)
        var best = emptyList<Int>()
        var bestScore = Double.NEGATIVE_INFINITY
        repeat(starts) {
            var selected = initial.shuffled(random).take(size)
            var current = score(selected)
            repeat(passes) {
                for (slot in 0 until size) {
                    for (replacement in initial.filter { it !in selected }.shuffled(random)) {
                        val proposal = selected.toMutableList().also { it[slot] = replacement }
                        val value = score(proposal)
                        if (value > current ||
                            (value == current && random.nextBoolean())
                        ) {
                            selected = proposal
                            current = value
                        }
                    }
                }
            }
            var radius = policy.initialRadius
            var plateau = 0
            var improvements = 0
            for (step in 0 until localSteps) {
                val previous = current
                for (slot in 0 until size) {
                    val candidates = neighborhood.around(
                        pool[selected[slot]].sample,
                        assumptions,
                        random,
                        radius,
                        neighbors,
                        anchors + pool.map { it.sample },
                    )
                    for (sample in candidates) {
                        val replacement = include(sample) ?: continue
                        if (replacement in selected) continue
                        val proposal = selected.toMutableList().also { it[slot] = replacement }
                        val value = score(proposal)
                        if (value > current ||
                            (value == current && random.nextBoolean())
                        ) {
                            selected = proposal
                            current = value
                        }
                    }
                }
                if (current > previous) {
                    plateau = 0
                    improvements++
                    if (policy.adaptive && improvements >= 3) {
                        radius = minOf(
                            policy.maxRadius,
                            radius * 2.0,
                        )
                        improvements = 0
                    }
                } else {
                    plateau++
                    improvements = 0
                    if (policy.adaptive && plateau % 2 == 0) radius = maxOf(policy.minRadius, radius * 0.5)
                    if (plateau >= policy.plateauSteps) break
                }
                if (!policy.adaptive) radius = policy.initialRadius / (step + 2.0)
            }
            if (best.isEmpty() || current > bestScore || (current == bestScore && random.nextBoolean())) {
                best = selected.toList()
                bestScore = current
            }
        }
        return JointCandidateSelection(
            best.map { pool[it] },
            if (size == count) WitnessTermination.CountReached else shortTermination,
            cache.size,
        )
    }
}
