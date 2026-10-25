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
    ): JointCandidateSelection {
        require(count > 0)
        val space = projection.space
        val assumptions = space.assumptionsFor(request.context)
        val validator = WitnessValidator(space)
        val excluded = request.excluded.map { it.assignmentKey() }.toSet()
        val generated = witnesses.witnesses(
            WitnessRequest(
                maxOf(candidateCount, count),
                assumptions,
                anchors = request.anchors,
                exclude = request.excluded,
                maxAttempts = (maxOf(candidateCount, count).toLong() * 2).coerceAtMost(Int.MAX_VALUE.toLong()).toInt(),
            ),
            random,
        )
        val pool = (request.anchors + generated.witnesses).distinctBy { it.assignmentKey() }.filter {
            it.assignmentKey() !in excluded && space.matches(it, assumptions) && validator.valid(it)
        }.map(::BanditSample)
        val shortTermination = if (generated.termination == WitnessTermination.CountReached) {
            WitnessTermination.BudgetExhausted
        } else {
            generated.termination
        }
        if (pool.isEmpty()) return JointCandidateSelection(emptyList(), shortTermination, 0)
        val size = minOf(count, pool.size)
        val cache = mutableMapOf<List<Int>, Double>()
        fun score(set: List<Int>): Double = cache.getOrPut(set.sorted()) {
            acquisition.evaluate(
                featureRows(set.sorted().map { projection.encode(pool[it], request.context) }, projection.featureSize),
            )
                .also { require(it.isFinite()) }
        }
        var best = emptyList<Int>()
        var bestScore = Double.NEGATIVE_INFINITY
        repeat(starts) {
            var selected = pool.indices.shuffled(random).take(size)
            var current = score(selected)
            repeat(passes) {
                for (slot in 0 until size) {
                    val alternatives = pool.indices.filter { it !in selected }.shuffled(random)
                    for (replacement in alternatives) {
                        val proposal = selected.toMutableList().also { it[slot] = replacement }
                        val value = score(proposal)
                        if (value > current || (value == current && random.nextBoolean())) {
                            selected = proposal
                            current = value
                        }
                    }
                }
            }
            if (current > bestScore || (current == bestScore && random.nextBoolean())) {
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
