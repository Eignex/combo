package com.eignex.combo.bo

import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.DecisionSpaceDef
import com.eignex.combo.decisions.FeatureEncoder
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.combo.shared.LearnerData
import com.eignex.combo.shared.SlotRemap
import com.eignex.combo.shared.WitnessSearch
import com.eignex.koblas.Vector
import com.eignex.kumulant.core.SeriesStat
import com.eignex.kumulant.stat.regression.tree.BatchForestConfig

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
    /** Candidate witnesses considered per acquisition search. */
    val candidateCount: Int = 64,
    /** Maximum proposal and fallback calls per acquisition search. */
    val maxWitnessAttempts: Int = 128,
    /** Acquisition-guided neighborhood iterations after random initialization. */
    val localSteps: Int = 8,
    /** Candidate neighbors attempted per iteration. */
    val neighbors: Int = 12,
) {
    init {
        require(
            candidateCount > 0 && maxWitnessAttempts >= 0 && localSteps >= 0 && neighbors > 0,
        ) { "invalid witness search limits" }
        require(initialRandomObservations >= 0) { "initialRandomObservations must be nonnegative" }
        require(randomExploration in 0.0..1.0) { "randomExploration must be a probability" }
        require(improvementMargin >= 0.0 && improvementMargin.isFinite()) { "invalid improvement margin" }
    }
}

/**
 * Forest convenience facade for the composable Bayesian optimization loop.
 * Kumulant supplies numeric CART fitting; Combo's generic components own lifecycle and search.
 *
 * @param projection Fixed model feature encoder.
 * @param store Observation store owned by the new study.
 * @property config Forest, exploration, and acquisition search configuration.
 * @property search Batch witness initialization provider.
 * @param randomSeed Reproducible fitting and round seed.
 * @param maximize Reward maximization or cost minimization.
 * @param rewards Optional weighted reward sink.
 * @param trainAbsError Optional post-fit error sink.
 * @param testAbsError Optional pre-fit error sink.
 * @param batchPolicy Pending exclusion or temporary fantasy batching.
 * @param outcome Objective noise and response normalization policy.
 */
class RandomForestOptimizer(
    projection: FeatureEncoder<Vector>,
    store: ObservationStore = InMemoryObservationStore(),
    val config: ForestOptimizerConfig = ForestOptimizerConfig(),
    val search: WitnessSearch = HybridDescentSession.klause(
        projection.space,
        candidateBatchSize = 64,
        poolCapacity = 256,
    ),
    randomSeed: Int = 0,
    maximize: Boolean = true,
    rewards: SeriesStat<*>? = null,
    trainAbsError: SeriesStat<*>? = null,
    testAbsError: SeriesStat<*>? = null,
    batchPolicy: BatchPolicy = SequentialBatchPolicy(),
    outcome: OutcomeSpec = OutcomeSpec(maximize = maximize),
) : BayesianOptimizer(
    projection = projection,
    surrogate = ForestSurrogate(config.forest),
    study = Study(projection.space, store),
    acquisitionFactory = ExpectedImprovement(config.improvementMargin),
    acquisitionOptimizer = LocalAndRandomSearch(
        projection,
        search,
        config.candidateCount,
        config.maxWitnessAttempts,
        config.localSteps,
        config.neighbors,
    ),
    batchPolicy = batchPolicy,
    outcome = outcome,
    initialDesign = RandomInitialDesign(config.initialRandomObservations, config.randomExploration),
    bootstrapThompson = config.acquisition == ForestAcquisition.BootstrapThompson,
    randomSeed = randomSeed,
    rewards = rewards,
    trainAbsError = trainAbsError,
    testAbsError = testAbsError,
) {
    init {
        require(outcome.maximize == maximize) { "objective directions differ" }
    }

    /** Factories retaining the compact forest API while using generic BO components. */
    companion object {
        /** Construct a forest/EI optimizer with normalized BO features. */
        fun smac(
            space: CompiledDecisionSpace,
            store: ObservationStore = InMemoryObservationStore(),
            config: ForestOptimizerConfig = ForestOptimizerConfig(),
            randomSeed: Int = 0,
            maximize: Boolean = true,
        ): RandomForestOptimizer = RandomForestOptimizer(
            SearchTransform(space),
            store,
            config.copy(acquisition = ForestAcquisition.ExpectedImprovement),
            randomSeed = randomSeed,
            maximize = maximize,
        )

        /** Construct independent history-bootstrap Thompson models per suggested batch slot. */
        fun bootstrapThompson(
            space: CompiledDecisionSpace,
            store: ObservationStore = InMemoryObservationStore(),
            config: ForestOptimizerConfig = ForestOptimizerConfig(),
            randomSeed: Int = 0,
            maximize: Boolean = true,
        ): RandomForestOptimizer = RandomForestOptimizer(
            SearchTransform(space),
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
 * @property trials Trial snapshots including pending reservations.
 * @property nextTrialId Next reservation identifier.
 * @property round Persisted suggestion random-stream position.
 */
class HistoryData internal constructor(
    val definition: DecisionSpaceDef,
    val history: ObservationHistory,
    /** Trial snapshots, including outstanding reservations. */
    val trials: List<Trial> = emptyList(),
    /** Next reservation identifier, preserved across state transfer. */
    val nextTrialId: Long = 1L,
    val round: Long = 0L,
) : LearnerData {
    override fun remap(slots: SlotRemap): LearnerData = throw UnsupportedOperationException(
        "History coordinates cannot be remapped; re-encode evaluations against the new schema.",
    )
}
