package com.eignex.combo.bandit.glm

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.combo.shared.LinearFeatureProjection
import com.eignex.combo.shared.NoFeasibleSampleException
import com.eignex.combo.shared.PredictionLearner
import com.eignex.combo.shared.RegressionLearning
import com.eignex.combo.shared.WitnessRequest
import com.eignex.combo.util.RandomSequence
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.solver.Sample
import com.eignex.klause.solver.objective.LinearObjective
import com.eignex.koblas.Vector
import com.eignex.koblas.dot
import com.eignex.kumulant.core.RegressionStat
import com.eignex.kumulant.core.SeriesStat
import com.eignex.kumulant.stat.regression.glm.LinearPosterior
import com.eignex.kumulant.stat.regression.glm.LinearRegressionResult
import kotlin.random.Random

/**
 * Linear-model bandit with posterior sampling over the weights.
 *
 * Each `choose` draws a weight vector from [posterior] given the current snapshot
 * of [regression], asks the [innerOptimizer] for the best-scoring feasible [Sample]
 * under those weights *given the current [Context]*, and yields it. Reward
 * observations [train] the underlying regression stat using the same
 * `(Sample, Context) → feature` projection.
 *
 * Nonlinear parameter scales use shared batch witness search to rank transformed
 * feature scores; [innerOptimizer] is used only for linear source objectives.
 *
 * Pair a [regression] flavour with a matching [posterior]:
 *  - [com.eignex.kumulant.stat.regression.glm.StochasticRegressionStat] + [com.eignex.kumulant.stat.regression.glm.PointPosterior]
 *  - [com.eignex.kumulant.stat.regression.glm.DiagonalRegressionStat] + [com.eignex.kumulant.stat.regression.glm.FactorisedGaussian]
 *  - [com.eignex.kumulant.stat.regression.glm.BayesianRegressionStat] + [com.eignex.kumulant.stat.regression.glm.MultivariateGaussian]
 */
class LinearBandit<R : LinearRegressionResult>(
    val projection: LinearFeatureProjection,
    val regression: RegressionStat<R>,
    val posterior: LinearPosterior<R>,
    val exploration: Double = 1.0,
    val innerOptimizer: (LinearObjective, Assumptions) -> Sample?,
    override val randomSeed: Int = System.currentTimeMillis().toInt(),
    override val maximize: Boolean = true,
    override val rewards: SeriesStat<*>? = null,
    override val trainAbsError: SeriesStat<*>? = null,
    override val testAbsError: SeriesStat<*>? = null,
) : PredictionLearner<LinearLearnerData> {

    val space get() = projection.space

    init {
        require(regression.featureSize == projection.featureSize) {
            "regression has ${regression.featureSize} features, projection has ${projection.featureSize}."
        }
    }

    private val scaledSearch by lazy { HybridDescentSession.klause(space, candidateBatchSize = 64, poolCapacity = 256) }

    private fun scaledChoice(weights: Vector, context: Context, random: Random): BanditSample {
        val batch = scaledSearch.witnesses(WitnessRequest(64, projection.assumptionsFor(context)), random)
        val sign = if (maximize) 1.0 else -1.0
        val sample = batch.witnesses.maxByOrNull { raw ->
            val features = projection.encode(BanditSample(raw), context)
            sign * (weights dot features)
        } ?: throw NoFeasibleSampleException("scaled candidate search returned no feasible witness")
        return BanditSample(sample)
    }

    private val randomSequence = RandomSequence(randomSeed)

    private val learning = RegressionLearning(
        regression,
        projection::encode,
        { features -> regression.read().predict(features) },
        rewards,
        trainAbsError,
        testAbsError,
    )

    fun predict(sample: BanditSample, context: Context): Double = learning.predict(sample, context)

    fun train(sample: BanditSample, context: Context, reward: Double, weight: Double = 1.0) =
        learning.train(sample, context, reward, weight)

    fun update(sample: BanditSample, context: Context, reward: Double, weight: Double = 1.0) =
        learning.update(sample, context, reward, weight)

    fun chooseOrThrow(context: Context): BanditSample {
        val rng = randomSequence.next()
        val snapshot = learning.snapshot()
        val sampledWeights = posterior.sample(snapshot, rng, exploration)
        if (projection.hasNonlinearScales) return scaledChoice(sampledWeights, context, rng)
        val objective = projection.toObjective(sampledWeights, snapshot.bias, context, maximize)
        val assumptions = projection.assumptionsFor(context)
        val raw = innerOptimizer(objective, assumptions)
            ?: throw NoFeasibleSampleException("inner optimizer returned no feasible sample")
        return BanditSample(raw)
    }

    fun choose(context: Context): BanditSample? = try {
        chooseOrThrow(context)
    } catch (_: NoFeasibleSampleException) {
        null
    }

    fun optimalOrThrow(context: Context): BanditSample {
        val snapshot = learning.snapshot()
        if (projection.hasNonlinearScales) return scaledChoice(snapshot.weights, context, randomSequence.next())
        val objective = projection.toObjective(snapshot.weights, snapshot.bias, context, maximize)
        val assumptions = projection.assumptionsFor(context)
        val raw = innerOptimizer(objective, assumptions)
            ?: throw NoFeasibleSampleException("inner optimizer returned no feasible sample")
        return BanditSample(raw)
    }

    fun optimal(context: Context): BanditSample? = try {
        optimalOrThrow(context)
    } catch (_: NoFeasibleSampleException) {
        null
    }

    // PredictionLearner interface — contextless overloads default to empty context.
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
    override fun chooseOrThrow(): BanditSample = chooseOrThrow(Context.Empty)
    override fun optimalOrThrow(): BanditSample = optimalOrThrow(Context.Empty)

    override fun importData(data: LinearLearnerData) {
        @Suppress("UNCHECKED_CAST")
        learning.merge(data.state as R)
    }

    override fun exportData(): LinearLearnerData = LinearLearnerData(learning.snapshot())
}
