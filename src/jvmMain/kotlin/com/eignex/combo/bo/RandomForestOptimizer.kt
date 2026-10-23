package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.DecisionSpaceDef
import com.eignex.combo.decisions.FeatureEncoder
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.combo.shared.LearnerData
import com.eignex.combo.shared.LinearFeatureProjection
import com.eignex.combo.shared.NoFeasibleSampleException
import com.eignex.combo.shared.PredictionLearner
import com.eignex.combo.shared.SlotRemap
import com.eignex.combo.util.RandomSequence
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.solver.Sample
import com.eignex.koblas.Vector
import com.eignex.kumulant.core.SeriesStat
import com.eignex.kumulant.stat.regression.tree.BatchForestConfig
import com.eignex.kumulant.stat.regression.tree.BatchRandomForest
import com.eignex.kumulant.stat.regression.tree.ForestRegressionResult
import com.eignex.kumulant.stat.regression.tree.expectedImprovement
import com.eignex.kumulant.stat.regression.tree.predictDistribution
import kotlin.random.Random

/** Acquisition rules for a history-fitted random forest. */
enum class ForestAcquisition {
    /** SMAC-style Gaussian expected improvement using the ensemble's tree predictions. */
    ExpectedImprovement,

    /** Refit a forest on a bootstrap of the full history and optimize its mean prediction. */
    BootstrapThompson,
}

/** Configuration for history-fitted forest optimization. */
data class ForestOptimizerConfig(
    /** Batch CART fitting configuration. */
    val forest: BatchForestConfig = BatchForestConfig(),
    /** Acquisition rule used for exploratory choices. */
    val acquisition: ForestAcquisition = ForestAcquisition.ExpectedImprovement,
    /** Number of history observations required before model-driven selection. */
    val initialRandomObservations: Int = 5,
    /** Probability of a random feasible choice after the initial design. */
    val randomExploration: Double = 0.1,
    /** Additional improvement required by expected improvement. */
    val improvementMargin: Double = 0.0,
) {
    init {
        require(initialRandomObservations >= 0) { "initialRandomObservations must be nonnegative" }
        require(randomExploration in 0.0..1.0) { "randomExploration must be a probability" }
        require(improvementMargin >= 0.0 && improvementMargin.isFinite()) { "invalid improvement margin" }
    }
}

/**
 * History-based forest optimization over feasible Klause configurations.
 *
 * [ForestAcquisition.ExpectedImprovement] refits on the full history when the store
 * revision changes, then ranks candidates by Gaussian expected improvement. This is
 * the SMAC model/acquisition pattern with Klause candidate search; racing, censored
 * trials, multi-fidelity scheduling, and SMAC's reference acquisition maximizer are
 * outside this optimizer. [ForestAcquisition.BootstrapThompson] instead bootstraps
 * the history and fits a fresh forest per choice, keeping that reward model fixed
 * throughout the candidate search.
 *
 * Initial and randomly interleaved exploration protects against collapsed empirical
 * forest uncertainty. Context is fixed during search and all evaluated contexts remain
 * in the training history. Calls are single-threaded. The store is bound to one schema.
 *
 * @property projection Numeric feature encoder; continuous values retain their declared units.
 * @property store Complete observation history, including off-policy training evaluations.
 * @property config Forest and acquisition configuration.
 * @property search Feasible candidate generation, witness caching, and repair.
 * @property randomSeed Reproducible fitting and candidate-search seed.
 * @property maximize True for reward maximization; false for cost minimization.
 * @property rewards Optional sink for observed rewards.
 * @property trainAbsError Optional absolute-error sink evaluated after recording each evaluation.
 * @property testAbsError Optional absolute-error sink evaluated before recording each evaluation.
 */
class RandomForestOptimizer(
    val projection: FeatureEncoder<Vector>,
    val store: ObservationStore = InMemoryObservationStore(),
    val config: ForestOptimizerConfig = ForestOptimizerConfig(),
    val search: HybridDescentSession = HybridDescentSession.klause(
        projection.space,
        candidateBatchSize = 64,
        poolCapacity = 256,
    ),
    override val randomSeed: Int = 0,
    override val maximize: Boolean = true,
    override val rewards: SeriesStat<*>? = null,
    override val trainAbsError: SeriesStat<*>? = null,
    override val testAbsError: SeriesStat<*>? = null,
) : PredictionLearner<HistoryData> {
    /** Decision space whose contextual variables are pinned during selection. */
    val space: CompiledDecisionSpace get() = projection.space

    private val fitter = BatchRandomForest(projection.featureSize, config.forest)
    private val random = RandomSequence(randomSeed)
    private val candidates = HistoryCandidates(space)
    private val learning = HistoryLearning(
        projection,
        store,
        fitter::fit,
        { model, features -> model.predictDistribution(features).mean },
        randomSeed,
        rewards,
        trainAbsError,
        testAbsError,
    )

    /** Number of full forest fits, including bootstrap draws. */
    val fitCount: Long get() = learning.fitCount

    override fun chooseOrThrow(): BanditSample = chooseOrThrow(Context.Empty)
    override fun optimalOrThrow(): BanditSample = optimalOrThrow(Context.Empty)
    override fun predict(sample: BanditSample): Double = predict(sample, Context.Empty)
    override fun train(sample: BanditSample, reward: Double, weight: Double) = train(
        sample,
        Context.Empty,
        reward,
        weight,
    )
    override fun update(sample: BanditSample, reward: Double, weight: Double) = update(
        sample,
        Context.Empty,
        reward,
        weight,
    )

    /** Choose a feasible configuration under the supplied context using the configured acquisition. */
    fun chooseOrThrow(context: Context): BanditSample {
        val assumptions = space.assumptionsFor(context)
        val rng = random.next()
        val history = store.snapshot()
        if (history.observations.isEmpty() || history.observations.size < config.initialRandomObservations ||
            rng.nextDouble() < config.randomExploration
        ) {
            return select(rng, assumptions, randomScores(rng))
        }
        if (config.acquisition == ForestAcquisition.BootstrapThompson) {
            val model = learning.fit(history, rng, bootstrap = true)
            return select(rng, assumptions) { signed(prediction(model, it, context)) }
        }
        val model = learning.model(history)
        val incumbent = incumbent(model, history, context, assumptions, rng)
        return select(rng, assumptions) { sample ->
            model.predictDistribution(projection.encode(BanditSample(sample), context))
                .expectedImprovement(incumbent, maximize, config.improvementMargin)
        }
    }

    /** Choose with context, returning null if candidate search yields no feasible witness. */
    fun choose(context: Context): BanditSample? = try {
        chooseOrThrow(context)
    } catch (_: NoFeasibleSampleException) {
        null
    }

    /** Recommend using a forest fitted to the full history and its mean prediction. */
    fun optimalOrThrow(context: Context): BanditSample {
        val model = learning.model()
        return select(random.next(), space.assumptionsFor(context)) { signed(prediction(model, it, context)) }
    }

    /** Recommend with context, returning null when no witness is found within the budgets. */
    fun optimal(context: Context): BanditSample? = try {
        optimalOrThrow(context)
    } catch (_: NoFeasibleSampleException) {
        null
    }

    /** Predict the mean from a forest fitted on the complete current history. */
    fun predict(sample: BanditSample, context: Context): Double = learning.predict(sample, context)

    /** Append an off-policy evaluation without updating reward or error sinks. */
    fun train(sample: BanditSample, context: Context, reward: Double, weight: Double = 1.0) =
        learning.train(sample, context, reward, weight)

    /** Append an evaluation and record the weighted reward and pre/post-fit errors. */
    fun update(sample: BanditSample, context: Context, reward: Double, weight: Double = 1.0) =
        learning.update(sample, context, reward, weight)

    override fun exportData(): HistoryData = HistoryData(space.definition, store.snapshot())

    override fun importData(data: HistoryData) {
        require(data.definition === space.definition) { "history belongs to a different decision-space definition" }
        for (observation in data.history.observations) store.append(observation)
    }

    private fun incumbent(
        model: ForestRegressionResult,
        history: ObservationHistory,
        context: Context,
        assumptions: Assumptions,
        rng: Random,
    ): Double {
        val matching = history.observations.filter {
            space.matches(
                it.sample.sample,
                assumptions,
            ) && candidates.valid(it.sample.sample)
        }
        if (matching.isNotEmpty()) {
            return if (maximize) matching.maxOf { it.reward } else matching.minOf { it.reward }
        }
        val reference = select(rng, assumptions) { signed(prediction(model, it, context)) }
        return prediction(model, reference.sample, context)
    }

    private fun prediction(model: ForestRegressionResult, sample: Sample, context: Context): Double =
        model.predictDistribution(projection.encode(BanditSample(sample), context)).mean

    private fun select(rng: Random, assumptions: Assumptions, score: (Sample) -> Double): BanditSample {
        search.beginRound(score)
        val generated = search.search(rng, assumptions).assignment
        val observed = candidates.best(store.snapshot(), assumptions, score)
        val sample = listOfNotNull(generated, observed).maxByOrNull(score)
            ?: throw NoFeasibleSampleException("candidate search returned no feasible witness within its budgets")
        return BanditSample(sample)
    }

    private fun randomScores(rng: Random): (Sample) -> Double {
        val scores = mutableMapOf<Sample, Double>()
        return { sample -> scores.getOrPut(sample) { rng.nextDouble() } }
    }

    private fun signed(value: Double): Double = if (maximize) value else -value

    /** Factories for the two history-based forest acquisition rules. */
    companion object {
        /** Construct a SMAC-style forest optimizer with expected improvement. */
        fun smac(
            space: CompiledDecisionSpace,
            store: ObservationStore = InMemoryObservationStore(),
            config: ForestOptimizerConfig = ForestOptimizerConfig(),
            randomSeed: Int = 0,
            maximize: Boolean = true,
        ): RandomForestOptimizer = RandomForestOptimizer(
            LinearFeatureProjection(space),
            store,
            config.copy(acquisition = ForestAcquisition.ExpectedImprovement),
            randomSeed = randomSeed,
            maximize = maximize,
        )

        /** Construct bootstrap Thompson sampling using a freshly rebuilt forest for each choice. */
        fun bootstrapThompson(
            space: CompiledDecisionSpace,
            store: ObservationStore = InMemoryObservationStore(),
            config: ForestOptimizerConfig = ForestOptimizerConfig(),
            randomSeed: Int = 0,
            maximize: Boolean = true,
        ): RandomForestOptimizer = RandomForestOptimizer(
            LinearFeatureProjection(space),
            store,
            config.copy(acquisition = ForestAcquisition.BootstrapThompson),
            randomSeed = randomSeed,
            maximize = maximize,
        )
    }
}

/**
 * Complete, schema-bound evaluation history for state transfer between optimizers.
 *
 * @property definition Exact schema definition to which stored solver coordinates belong.
 * @property history Stable snapshot of the stored evaluations.
 */
class HistoryData internal constructor(val definition: DecisionSpaceDef, val history: ObservationHistory) :
    LearnerData {
    override fun remap(slots: SlotRemap): LearnerData = throw UnsupportedOperationException(
        "History coordinates cannot be remapped; re-encode evaluations against the new schema.",
    )
}
