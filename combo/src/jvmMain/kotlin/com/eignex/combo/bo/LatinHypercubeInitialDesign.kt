package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.combo.shared.WitnessRequest
import com.eignex.combo.shared.WitnessSearch
import com.eignex.combo.shared.assignmentKey
import com.eignex.koblas.DenseVector
import kotlin.random.Random

/**
 * Reproducible continuous Latin hypercube, sized by decision dimension and external evaluation budget.
 * Rejected constrained draws fall back to independently verified feasible witnesses. Stratification
 * describes the unconstrained design; rejection and fallback do not preserve uniform constrained coverage.
 * Plan positions derive from recorded observations and issued trials, so state transfer needs no hidden cursor.
 *
 * @property space Original decision space.
 * @property evaluationBudget Maximum intended external evaluations, used to cap the initial design.
 * @property perDimension Requested initial observations per nonconstant decision dimension.
 * @property maxRatio Maximum budget fraction devoted to initial design, with at least one initial point.
 * @property probability Independent exploration probability after the initial design.
 * @property seed Plan seed, independent of model fitting and per-round search randomness.
 * @property witnesses Feasible anchors and fallback proposals.
 * @property measurementsPerConfiguration Planned measurement allowance per initial configuration.
 */
class LatinHypercubeInitialDesign(
    val space: CompiledDecisionSpace,
    val evaluationBudget: Int,
    val perDimension: Int = 10,
    val maxRatio: Double = 0.25,
    val probability: Double = 0.2,
    val seed: Int = 0,
    val witnesses: WitnessSearch = HybridDescentSession.klause(space),
    val measurementsPerConfiguration: Double = 1.0,
) : InitialDesign {
    private val transform = SearchTransform(space)

    /** Number of planned initial configurations. */
    val size: Int = minOf(
        (perDimension.toLong() * maxOf(1, transform.dimension)).coerceAtMost(Int.MAX_VALUE.toLong()).toInt(),
        maxOf(1, (evaluationBudget * maxRatio / measurementsPerConfiguration).toInt()),
    )
    private val plan: List<DoubleArray> by lazy {
        val random = Random(seed)
        val rows = List(size) { DoubleArray(transform.featureSize) }
        for (column in 0 until transform.featureSize) {
            val strata = (0 until size).shuffled(random)
            for (row in rows.indices) rows[row][column] = (strata[row] + random.nextDouble()) / size
        }
        rows
    }
    init {
        require(evaluationBudget > 0 && perDimension > 0)
        require(measurementsPerConfiguration >= 1.0 && measurementsPerConfiguration.isFinite())
        require(maxRatio > 0.0 && maxRatio <= 1.0 && probability in 0.0..1.0)
    }
    private fun position(snapshot: StudySnapshot): Int {
        val issued = snapshot.trials.map { it.sample.sample.assignmentKey() }.toSet()
        val offPolicy = snapshot.history.observations.map { it.sample.sample.assignmentKey() }.toSet() - issued
        return snapshot.trials.size + offPolicy.size
    }
    override fun explore(snapshot: StudySnapshot, random: Random): Boolean =
        position(snapshot) < size || snapshot.history.observations.isEmpty() || random.nextDouble() < probability

    override fun sample(request: InitialDesignRequest, random: Random): CandidateSelection? {
        val index = position(request.snapshot) + request.offset
        if (index >= size) return null
        val batch = witnesses.witnesses(
            WitnessRequest(1, space.assumptionsFor(request.context), exclude = request.excluded),
            random,
        )
        val anchor = batch.witnesses.firstOrNull() ?: return CandidateSelection(null, batch.termination, 0)
        val point = transform.sampleUnit(DenseVector.of(plan[index]), BanditSample(anchor), request.context)
        val excluded = request.excluded.map { it.assignmentKey() }.toSet()
        val chosen = point?.takeIf { it.sample.assignmentKey() !in excluded } ?: BanditSample(anchor)
        return CandidateSelection(chosen, batch.termination, 0)
    }
}
