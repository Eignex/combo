package combo.bandit.glm

import com.eignex.klause.solver.Assumptions
import com.eignex.klause.solver.LinearObjective
import com.eignex.klause.solver.Sample
import com.eignex.kumulant.stat.regression.glm.LinearPosterior
import com.eignex.kumulant.core.RegressionStat
import com.eignex.kumulant.core.SeriesStat
import com.eignex.kumulant.stat.regression.glm.LinearRegressionResult
import combo.bandit.NoFeasibleSampleException
import combo.bandit.PredictionLearner
import combo.decisions.BanditSample
import combo.decisions.Context
import combo.util.RandomSequence
import kotlin.math.abs

/**
 * Linear-model bandit with posterior sampling over the weights.
 *
 * Each `choose` draws a weight vector from [posterior] given the current snapshot
 * of [regression], asks the [innerOptimizer] for the best-scoring feasible [Sample]
 * under those weights *given the current [Context]*, and yields it. Reward
 * observations [train] the underlying regression stat using the same
 * `(Sample, Context) → feature` projection.
 *
 * Pair a [regression] flavour with a matching [posterior]:
 *  - [com.eignex.kumulant.stat.regression.SGDLinearRegression] + [com.eignex.kumulant.stat.regression.PointPosterior]
 *  - [com.eignex.kumulant.stat.regression.DiagonalRegression] + [com.eignex.kumulant.stat.regression.FactorisedGaussian]
 *  - [com.eignex.kumulant.stat.regression.BayesianLinearRegression] + [com.eignex.kumulant.stat.regression.MultivariateGaussian]
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

    private val randomSequence = RandomSequence(randomSeed)

    fun predict(sample: BanditSample, context: Context): Double {
        val snapshot = regression.read()
        return snapshot.predict(projection.encode(sample, context))
    }

    fun train(sample: BanditSample, context: Context, reward: Double, weight: Double = 1.0) {
        regression.update(projection.encode(sample, context), reward, weight)
    }

    fun update(sample: BanditSample, context: Context, reward: Double, weight: Double = 1.0) {
        @Suppress("UNCHECKED_CAST")
        (rewards as? SeriesStat<Any>)?.update(reward, 0L, weight)
        if (testAbsError != null) {
            val err = abs(reward - predict(sample, context))
            @Suppress("UNCHECKED_CAST")
            (testAbsError as SeriesStat<Any>).update(err, 0L, weight)
        }
        train(sample, context, reward, weight)
        if (trainAbsError != null) {
            val err = abs(reward - predict(sample, context))
            @Suppress("UNCHECKED_CAST")
            (trainAbsError as SeriesStat<Any>).update(err, 0L, weight)
        }
    }

    fun chooseOrThrow(context: Context): BanditSample {
        val rng = randomSequence.next()
        val snapshot = regression.read()
        val sampledWeights = posterior.sample(snapshot, rng, exploration)
        val objective = projection.toObjective(sampledWeights, snapshot.bias, context, maximize)
        val assumptions = projection.assumptionsFor(context)
        val raw = innerOptimizer(objective, assumptions)
            ?: throw NoFeasibleSampleException("inner optimizer returned no feasible sample")
        return BanditSample.dithered(raw, space, rng)
    }

    fun choose(context: Context): BanditSample? = try {
        chooseOrThrow(context)
    } catch (_: NoFeasibleSampleException) {
        null
    }

    fun optimalOrThrow(context: Context): BanditSample {
        val rng = randomSequence.next()
        val snapshot = regression.read()
        val objective = projection.toObjective(snapshot.weights, snapshot.bias, context, maximize)
        val assumptions = projection.assumptionsFor(context)
        val raw = innerOptimizer(objective, assumptions)
            ?: throw NoFeasibleSampleException("inner optimizer returned no feasible sample")
        return BanditSample.dithered(raw, space, rng)
    }

    fun optimal(context: Context): BanditSample? = try {
        optimalOrThrow(context)
    } catch (_: NoFeasibleSampleException) {
        null
    }

    // PredictionLearner interface — contextless overloads default to empty context.
    override fun predict(sample: BanditSample): Double = predict(sample, Context.Empty)
    override fun train(sample: BanditSample, reward: Double, weight: Double) = train(sample, Context.Empty, reward, weight)
    override fun update(sample: BanditSample, reward: Double, weight: Double) = update(sample, Context.Empty, reward, weight)
    override fun chooseOrThrow(): BanditSample = chooseOrThrow(Context.Empty)
    override fun optimalOrThrow(): BanditSample = optimalOrThrow(Context.Empty)

    override fun importData(data: LinearLearnerData) {
        @Suppress("UNCHECKED_CAST")
        (regression as RegressionStat<LinearRegressionResult>).merge(data.state)
    }

    override fun exportData(): LinearLearnerData = LinearLearnerData(regression.read())
}
