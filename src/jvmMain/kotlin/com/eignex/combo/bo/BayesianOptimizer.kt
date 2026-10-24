package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.FeatureEncoder
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.combo.shared.NoFeasibleSampleException
import com.eignex.combo.shared.PredictionLearner
import com.eignex.combo.shared.WitnessValidator
import com.eignex.combo.shared.assignmentKey
import com.eignex.koblas.DenseVector
import com.eignex.koblas.Vector
import com.eignex.koblas.dot
import com.eignex.koblas.sum
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
    /** Acquisition evaluations, excluding expensive external objective evaluations. */
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

/** Decide whether a round uses random initial/interleaved exploration. */
fun interface InitialDesign {
    /** Decide once per frozen batch, before fitting a surrogate. */
    fun explore(snapshot: StudySnapshot, random: Random): Boolean
}

/** Initial observations followed by randomly interleaved exploration. */
class RandomInitialDesign(
    /** Number of observations before model-driven search. */
    val observations: Int = 5,
    /** Exploration probability after initial design. */
    val probability: Double = 0.1,
) : InitialDesign {
    init {
        require(observations >= 0 && probability in 0.0..1.0)
    }
    override fun explore(snapshot: StudySnapshot, random: Random): Boolean =
        snapshot.history.observations.isEmpty() || snapshot.history.observations.size < observations ||
            random.nextDouble() < probability
}

/**
 * Composable history-based Bayesian optimization loop over feasible Klause configurations.
 * One study snapshot, response encoding, and base fitted model are shared by each batch.
 * Acquisition maximization, initial exploration, and temporary fantasy policy are replaceable.
 * Numeric inputs/outputs use Koblas directly; statistical forest fitting remains in Kumulant.
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
 * @property trainAbsError Optional post-recording prediction error sink.
 * @property testAbsError Optional pre-recording prediction error sink.
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
    override val trainAbsError: SeriesStat<*>? = null,
    override val testAbsError: SeriesStat<*>? = null,
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
        trainAbsError,
        testAbsError,
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
            space.matches(it.sample.sample, space.assumptionsFor(context)) && candidates.valid(it.sample.sample)
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
        val selector = object : BatchSelector {
            override fun predict(sample: BanditSample, context: Context): Double =
                base.mean(projection.encode(sample, context))
            override fun select(
                temporary: List<Observation>,
                excluded: List<com.eignex.klause.solver.Sample>,
            ): CandidateSelection {
                val acquisition = if (explore) {
                    val scores = mutableMapOf<List<Double>, Double>()
                    Acquisition { features ->
                        DenseVector.of(
                            DoubleArray(features.rows) { i ->
                                scores.getOrPut(features.rowVector(i).toDoubleArray().toList()) { random.nextDouble() }
                            },
                        )
                    }
                } else {
                    val fitted = when {
                        bootstrapThompson -> learning.fit(
                            snapshot.history.observations + temporary,
                            random,
                            bootstrap = true,
                            response = ResponseTransform.from(snapshot.history.observations, outcome.standardize),
                        )

                        temporary.isNotEmpty() -> learning.temporary(snapshot.history, temporary, random)

                        else -> base
                    }
                    val factory = if (bootstrapThompson) PosteriorMean else acquisitionFactory
                    factory.create(AcquisitionContext(fitted, baseline, outcome), random)
                }
                val selected = acquisitionOptimizer.maximize(
                    acquisition,
                    AcquisitionRequest(context, excluded, scoped.map { it.sample.sample }),
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
    fun recommendOrThrow(context: Context = Context.Empty): BanditSample {
        val snapshot = study.beginRound()
        val model = learning.model(snapshot.history)
        val acquisition = PosteriorMean.create(AcquisitionContext(model, null, outcome), Random(randomSeed))
        return acquisitionOptimizer.maximize(
            acquisition,
            AcquisitionRequest(
                context,
                emptyList(),
                snapshot.history.observations.map { it.sample.sample },
            ),
            Random(randomSeed.toLong() xor snapshot.round),
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

    /** Best actually evaluated configuration; noisy observations are aggregated by statistical weight. */
    fun bestObserved(context: Context = Context.Empty): ObservedBest? {
        val observations = store.snapshot().observations.filter {
            space.matches(it.sample.sample, space.assumptionsFor(context)) && candidates.valid(it.sample.sample)
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
    override fun importData(data: HistoryData) = study.importData(data)
}
