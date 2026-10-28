package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.FeatureEncoder
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.combo.shared.NoFeasibleSampleException
import com.eignex.combo.shared.PredictionLearner
import com.eignex.combo.shared.WitnessRequest
import com.eignex.combo.shared.WitnessSearch
import com.eignex.combo.shared.WitnessValidator
import com.eignex.combo.shared.assignmentKey
import com.eignex.koblas.DenseVector
import com.eignex.koblas.Vector
import com.eignex.koblas.dot
import com.eignex.koblas.sum
import com.eignex.kumulant.core.PairedStat
import com.eignex.kumulant.core.SeriesStat
import kotlin.random.Random

/** Reproducibility and acquisition-work diagnostics for the most recent suggestion round. */
data class SuggestionTrace(
    /** Persisted study round identifier. */
    val round: Long,
    /** Frozen observation revision. */
    val historyRevision: Long,
    /** Seed used for proposal, acquisition, and batch randomness. */
    val seed: Long,
    /** Full model fits performed in this round, including temporary bootstrap/fantasy fits. */
    val modelFits: Long,
    /** Acquisition objective evaluations (individual points or joint sets), excluding external evaluations. */
    val acquisitionEvaluations: Int,
)

/** Best evaluated configuration with an explicit aggregate for repeated noisy measurements. */
data class ObservedBest(
    /** Feasible original configuration. */
    val sample: BanditSample,
    /** Exact observed response, or weighted replicate mean under the noisy policy. */
    val reward: Double,
    /** Number of evaluations contributing to this recommendation. */
    val observations: Int,
)

/**
 * Composable history-based Bayesian optimization loop over feasible Klause configurations.
 * One study snapshot, response encoding, and base fitted model are shared by each batch.
 * Acquisition maximization, initial exploration, and temporary fantasy policy are replaceable.
 * Numeric inputs/outputs use Koblas directly; batch forest fitting lives in Combo’s model component.
 * Calls are single-threaded; external evaluations can complete in arbitrary order.
 *
 * @property projection Model feature encoding, fixed for this optimizer's lifetime.
 * @property surrogate Full-history numeric fitter.
 * @property study Lifecycle and observation owner.
 * @property acquisitionFactory Acquisition objective bound to each immutable model.
 * @property acquisitionOptimizer Feasible acquisition-guided search.
 * @property batchPolicy Pending evaluation and batch construction strategy.
 * @property outcome Objective orientation, noise interpretation, and response normalization.
 * @property initialDesign Initial and interleaved random exploration policy.
 * @property bootstrapThompson Fit a new history-bootstrap reward model for each batch slot.
 * @property randomSeed Seed combined with the exported study round counter.
 * @property rewards Optional weighted reward sink.
 * @property trainScore Optional paired scorer evaluated after recording an observation.
 * @property testScore Optional paired scorer evaluated before recording an observation.
 * @property repeatPolicy Whether evaluated assignments remain eligible for another suggestion.
 * @property exploration Independent feasible proposal source for random exploration rounds.
 */
open class BayesianOptimizer(
    val projection: FeatureEncoder<Vector>,
    val surrogate: Surrogate,
    val study: Study = Study(projection.space),
    val acquisitionFactory: AcquisitionFactory = ExpectedImprovement(),
    val acquisitionOptimizer: AcquisitionOptimizer = LocalAndRandomSearch(
        projection,
        HybridDescentSession.klause(projection.space, candidateBatchSize = 64, poolCapacity = 256),
    ),
    val batchPolicy: BatchPolicy = SequentialBatchPolicy(),
    val outcome: OutcomeSpec = OutcomeSpec(),
    val initialDesign: InitialDesign = RandomInitialDesign(),
    val bootstrapThompson: Boolean = false,
    override val randomSeed: Int = 0,
    override val rewards: SeriesStat<*>? = null,
    override val trainScore: PairedStat<*>? = null,
    override val testScore: PairedStat<*>? = null,
    val repeatPolicy: RepeatPolicy = RepeatPolicy.NoisyOnly,
    val exploration: WitnessSearch = HybridDescentSession.klause(projection.space),
) : PredictionLearner<HistoryData>,
    TrialOptimizer {
    /** Original decision space, used exclusively for feasibility and decoding. */
    val space get() = projection.space

    /** Observation storage owned by the study. */
    val store: ObservationStore get() = study.store
    override val maximize: Boolean get() = outcome.maximize
    override val trials: List<Trial> get() = study.trials

    /** Last completed algorithm round, including its reproducible seed and work counts. */
    var lastSuggestion: SuggestionTrace? = null
        private set

    private val candidates = WitnessValidator(space)
    private val learning = SurrogateLearning(
        projection,
        store,
        surrogate,
        outcome,
        randomSeed,
        rewards,
        trainScore,
        testScore,
    )

    /** Number of full fits, including temporary and metric-triggered refits. */
    val fitCount: Long get() = learning.fitCount

    init {
        require(study.space === projection.space) { "study and projection must use the same compiled space" }
    }

    override fun suggest(count: Int, context: Context): TrialBatch = study.suggest(
        SuggestionPolicy(::propose),
        count,
        context,
    )
    override fun complete(id: Long, reward: Double, weight: Double, noiseVariance: Double?) {
        study.completeWith(id, reward, weight, noiseVariance) { trial, committed ->
            learning.update(trial.sample, trial.context, reward, weight, noiseVariance, committed)
        }
    }
    override fun fail(id: Long, reason: String) = study.fail(id, reason)
    override fun cancel(id: Long, reason: String?) = study.cancel(id, reason)

    private fun propose(snapshot: StudySnapshot, count: Int, context: Context): ProposalBatch {
        val seed = randomSeed.toLong() xor (snapshot.round * -7046029254386353131L)
        val random = Random(seed)
        val beforeFits = fitCount
        var evaluations = 0
        val explore = initialDesign.explore(snapshot, random)
        val base by lazy { learning.model(snapshot.history) }
        val scoped = snapshot.history.observations.filter {
            space.matches(it.sample.sample, context) && candidates.valid(it.sample.sample)
        }
        val completed = if (repeatPolicy == RepeatPolicy.Never ||
            (repeatPolicy == RepeatPolicy.NoisyOnly && outcome.noise == ObservationNoise.Deterministic)
        ) {
            snapshot.history.observations.map { it.sample.sample }
        } else {
            emptyList()
        }
        val baseline by lazy {
            if (scoped.isEmpty()) {
                null
            } else if (outcome.noise == ObservationNoise.Noisy) {
                scoped.distinctBy { it.sample.sample.assignmentKey() }.map {
                    base.mean(projection.encode(it.sample, context))
                }.maxByOrNull(outcome::score)
            } else {
                scoped.map { it.reward }.maxByOrNull(outcome::score)
            }
        }
        var offset = 0
        val selector = object : BatchSelector {
            override fun predict(sample: BanditSample, context: Context): Double =
                base.mean(projection.encode(sample, context))
            override fun selectJoint(
                count: Int,
                acquisition: JointAcquisitionFactory,
                optimizer: JointAcquisitionOptimizer,
            ): ProposalBatch {
                require(!bootstrapThompson) { "joint batches do not support bootstrap Thompson sampling" }
                if (explore || scoped.isEmpty()) {
                    return SequentialBatchPolicy().select(snapshot, count, context, this)
                }
                val pending = snapshot.pending.filter {
                    space.matches(it.sample.sample, context) && candidates.valid(it.sample.sample)
                }
                val completedRows = scoped.distinctBy { it.sample.sample.assignmentKey() }
                val objective = acquisition.create(
                    JointAcquisitionContext(
                        base,
                        baseline,
                        outcome,
                        featureRows(
                            completedRows.map { projection.encode(it.sample, context) },
                            projection.featureSize,
                        ),
                        featureRows(pending.map { projection.encode(it.sample, context) }, projection.featureSize),
                    ),
                    random,
                )
                val selected = optimizer.maximize(
                    objective,
                    AcquisitionRequest(
                        context,
                        snapshot.pending.map { it.sample.sample } + completed,
                        scoped.map { it.sample.sample },
                    ),
                    count,
                    random,
                )
                evaluations += selected.evaluations
                return ProposalBatch(selected.samples, selected.termination)
            }
            override fun select(
                temporary: List<Observation>,
                excluded: List<com.eignex.klause.solver.Sample>,
            ): CandidateSelection {
                if (explore) {
                    val unavailable = excluded + completed
                    val designed = initialDesign.sample(
                        InitialDesignRequest(snapshot, context, unavailable, offset++),
                        random,
                    )
                    if (designed != null) return designed
                    val batch = exploration.witnesses(
                        WitnessRequest(
                            1,
                            space.discreteAssumptionsFor(context),
                            exclude = unavailable,
                            anchors = scoped.map { it.sample.sample },
                            context = context,
                        ),
                        random,
                    )
                    return CandidateSelection(batch.witnesses.firstOrNull()?.let(::BanditSample), batch.termination, 0)
                }
                val acquisition = run {
                    val fitted = when {
                        bootstrapThompson -> learning.fit(
                            snapshot.history.observations + temporary,
                            random,
                            bootstrap = true,
                            response = ResponseTransform.from(snapshot.history.observations, outcome),
                        )

                        temporary.isNotEmpty() -> learning.temporary(snapshot.history, temporary, random)

                        else -> base
                    }
                    val factory = if (bootstrapThompson) PosteriorMean else acquisitionFactory
                    factory.create(AcquisitionContext(fitted, baseline, outcome), random)
                }
                val selected = acquisitionOptimizer.maximize(
                    acquisition,
                    AcquisitionRequest(context, excluded + completed, scoped.map { it.sample.sample }),
                    random,
                )
                evaluations += selected.evaluations
                return selected
            }
        }
        val proposals = batchPolicy.select(snapshot, count, context, selector)
        lastSuggestion = SuggestionTrace(
            snapshot.round,
            snapshot.history.revision,
            seed,
            fitCount - beforeFits,
            evaluations,
        )
        return proposals
    }

    override fun chooseOrThrow(): BanditSample = chooseOrThrow(Context.Empty)

    /** Choose without reserving an evaluation, still excluding outstanding study trials. */
    fun chooseOrThrow(context: Context): BanditSample {
        val result = propose(study.beginRound(), 1, context)
        return result.samples.firstOrNull() ?: throw NoFeasibleSampleException(
            "acquisition search: ${result.termination}",
        )
    }

    /** Contextual nullable compatibility wrapper. */
    fun choose(context: Context): BanditSample? = try {
        chooseOrThrow(
            context,
        )
    } catch (_: NoFeasibleSampleException) {
        null
    }

    /** Recommend a feasible configuration by posterior mean, which may not have been evaluated. */
    fun recommendOrThrow(context: Context = Context.Empty): BanditSample =
        recommendOrThrow(acquisitionOptimizer, context)

    /**
     * Recommend using an independent search budget over the cached posterior mean.
     * Reads a snapshot without advancing suggestion randomness or reserving an evaluation.
     *
     * @param search Feasible maximizer used only for recommendation.
     * @param context Hard context shared by the recommendation candidates.
     */
    fun recommendOrThrow(search: AcquisitionOptimizer, context: Context = Context.Empty): BanditSample {
        val snapshot = study.snapshot()
        val model = learning.model(snapshot.history)
        val acquisition = PosteriorMean.create(AcquisitionContext(model, null, outcome), Random(randomSeed))
        return search.maximize(
            acquisition,
            AcquisitionRequest(
                context,
                emptyList(),
                snapshot.history.observations.map { it.sample.sample },
            ),
            Random(randomSeed.toLong() xor snapshot.history.revision),
        ).sample
            ?: throw NoFeasibleSampleException("recommendation search returned no feasible witness")
    }
    override fun optimalOrThrow(): BanditSample = recommendOrThrow()

    /** Compatibility alias for model recommendation; use bestObserved for evaluated configurations. */
    fun optimalOrThrow(context: Context): BanditSample = recommendOrThrow(context)

    /** Contextual nullable compatibility wrapper. */
    fun optimal(context: Context): BanditSample? = try {
        recommendOrThrow(
            context,
        )
    } catch (_: NoFeasibleSampleException) {
        null
    }

    internal fun fittedModel(): FittedSurrogate = learning.model(store.snapshot())

    /** Best actually evaluated configuration; noisy observations are aggregated by statistical weight. */
    fun bestObserved(context: Context = Context.Empty): ObservedBest? {
        val observations = store.snapshot().observations.filter {
            space.matches(it.sample.sample, context) && candidates.valid(it.sample.sample)
        }
        val groups = if (outcome.noise ==
            ObservationNoise.Noisy
        ) {
            observations.groupBy { it.sample.sample.assignmentKey() }.values
        } else {
            observations.map { listOf(it) }
        }
        return groups.map { group ->
            val weights = DenseVector.of(DoubleArray(group.size) { group[it].weight })
            val rewards = DenseVector.of(DoubleArray(group.size) { group[it].reward })
            ObservedBest(group.first().sample, (weights dot rewards) / weights.sum(), group.size)
        }.maxByOrNull { outcome.score(it.reward) }
    }

    override fun predict(sample: BanditSample): Double = predict(sample, Context.Empty)

    /** Mean prediction in original reward units, with a revision-cached full-history fit. */
    fun predict(sample: BanditSample, context: Context): Double = learning.predict(sample, context)
    override fun train(sample: BanditSample, reward: Double, weight: Double) = train(
        sample,
        Context.Empty,
        reward,
        weight,
    )

    /** Append an off-policy observation, independently specifying statistical weight and known noise variance. */
    fun train(
        sample: BanditSample,
        context: Context,
        reward: Double,
        weight: Double = 1.0,
        noiseVariance: Double? = null,
    ) = learning.train(sample, context, reward, weight, noiseVariance)
    override fun update(sample: BanditSample, reward: Double, weight: Double) = update(
        sample,
        Context.Empty,
        reward,
        weight,
    )

    /** Record an observation and reward/error metrics in original units. */
    fun update(
        sample: BanditSample,
        context: Context,
        reward: Double,
        weight: Double = 1.0,
        noiseVariance: Double? = null,
    ) = learning.update(sample, context, reward, weight, noiseVariance)
    override fun exportData(): HistoryData = study.exportData()
    override fun importData(data: HistoryData) {
        require(outcome.responseScale != ResponseScale.Log || data.history.observations.all { it.reward > 0.0 })
        study.importData(data)
    }
}
