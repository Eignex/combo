package com.eignex.combo.bandit.glm

import com.eignex.combo.bo.*
import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context
import com.eignex.combo.model.LinearLearning
import com.eignex.combo.shared.*
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.solver.Sample
import com.eignex.klause.solver.objective.LinearObjective
import com.eignex.kumulant.core.SeriesStat
import kotlin.random.Random

/** Native linear posterior sampling with the shared asynchronous study lifecycle. */
class LinearTrialOptimizer(
    /** Existing native feature projection; statistical inputs arrive as pinned context. */
    val projection: LinearFeatureProjection,
    learning: LinearLearning,
    private val search: WitnessSearch,
    innerOptimizer: (LinearObjective, Assumptions) -> Sample?,
    private val seed: Int = 0,
    maximize: Boolean = true,
    private val rewards: SeriesStat<*>? = null,
) : TrialOptimizer {
    init {
        require(!projection.hasNonlinearScales) { "linear trial proposals currently require linear scales" }
    }

    private val study = Study(projection.space)
    private var excluded: List<Sample> = emptyList()
    private var round = 0L
    private val learner = learning.materialize(projection, { objective, assumptions ->
        if (excluded.isEmpty()) innerOptimizer(objective, assumptions) else {
            search.witnesses(
                WitnessRequest(64, assumptions, exclude = excluded), Random(seed.toLong() xor round++),
            ).witnesses.minByOrNull(objective::evaluate)
        }
    }, seed, maximize)

    /** Successful outcomes, independent of statistical input observations. */
    val observationCount: Long get() = study.store.revision
    override val trials: List<Trial> get() = study.trials

    /** Predict using the persistent posterior mean and the supplied assignment's original context. */
    fun predict(sample: BanditSample, context: Context): Double = learner.predict(sample, context)

    /** Optimize the posterior mean with the unchanged native linear bandit. */
    fun recommend(context: Context): BanditSample = learner.optimalOrThrow(context)

    override fun suggest(count: Int, context: Context): TrialBatch = study.suggest({ snapshot, requested, given ->
        val samples = mutableListOf<BanditSample>()
        excluded = snapshot.pending.map { it.sample.sample }
        try {
            for (index in 0 until requested) {
                val sample = learner.choose(given) ?: break
                samples += sample
                excluded = excluded + sample.sample
            }
        } finally { excluded = emptyList() }
        val termination = if (samples.size == requested) WitnessTermination.CountReached else WitnessTermination.BudgetExhausted
        ProposalBatch(samples, termination)
    }, count, context)

    override fun complete(id: Long, reward: Double, weight: Double, noiseVariance: Double?) {
        require(noiseVariance == null) { "linear learning does not support per-observation noiseVariance" }
        study.completeWith(id, reward, weight, noiseVariance) { trial, committed ->
            learner.train(trial.sample, trial.context, reward, weight)
            study.store.append(Observation(trial.sample, trial.context, reward, weight))
            committed()
        }
        rewards?.update(reward, weight)
    }

    override fun fail(id: Long, reason: String) = study.fail(id, reason)
    override fun cancel(id: Long, reason: String?) = study.cancel(id, reason)
}
