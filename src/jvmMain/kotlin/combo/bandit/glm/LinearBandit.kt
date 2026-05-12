package combo.bandit.glm

import com.eignex.klause.solver.LinearObjective
import com.eignex.klause.solver.Sample
import com.eignex.kumulant.core.SeriesStat
import combo.bandit.NoFeasibleSampleException
import combo.bandit.PredictionBandit
import combo.decisions.Context
import combo.util.RandomSequence
import kotlin.math.abs

/**
 * Generalised linear model bandit with Thompson sampling over the model weights.
 *
 * Each `choose` draws a weight vector from [linearModel]'s posterior, asks the
 * [innerOptimizer] for the best-scoring feasible [Sample] under those weights *given
 * the current [Context]*, and yields it. Reward observations [train] the underlying
 * linear model using the same `(Sample, Context) → feature` projection.
 */
class LinearBandit(
    val projection: LinearFeatureProjection,
    val linearModel: LinearModel,
    val innerOptimizer: (LinearObjective) -> Sample?,
    override val randomSeed: Int = System.currentTimeMillis().toInt(),
    override val maximize: Boolean = true,
    override val rewards: SeriesStat<*>? = null,
    override val trainAbsError: SeriesStat<*>? = null,
    override val testAbsError: SeriesStat<*>? = null,
) : PredictionBandit<LinearData> {

    val space get() = projection.space

    init {
        require(linearModel.weights.size == projection.featureSize) {
            "Linear model has ${linearModel.weights.size} weights, projection has ${projection.featureSize} features."
        }
    }

    private val randomSequence = RandomSequence(randomSeed)

    fun predict(sample: Sample, context: Context): Double =
        linearModel.predict(projection.encode(sample, context)).toDouble()

    fun train(sample: Sample, context: Context, reward: Double, weight: Double = 1.0) {
        linearModel.train(projection.encode(sample, context), reward.toFloat(), weight.toFloat())
    }

    fun update(sample: Sample, context: Context, reward: Double, weight: Double = 1.0) {
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

    fun chooseOrThrow(context: Context): Sample {
        val rng = randomSequence.next()
        val sampledWeights = linearModel.sample(rng)
        val objective = projection.toObjective(sampledWeights, linearModel.bias, context, maximize)
        return innerOptimizer(objective)
            ?: throw NoFeasibleSampleException("inner optimizer returned no feasible sample")
    }

    fun choose(context: Context): Sample? = try {
        chooseOrThrow(context)
    } catch (_: NoFeasibleSampleException) {
        null
    }

    fun optimalOrThrow(context: Context): Sample {
        val objective = projection.toObjective(linearModel.weights, linearModel.bias, context, maximize)
        return innerOptimizer(objective)
            ?: throw NoFeasibleSampleException("inner optimizer returned no feasible sample")
    }

    fun optimal(context: Context): Sample? = try {
        optimalOrThrow(context)
    } catch (_: NoFeasibleSampleException) {
        null
    }

    // PredictionBandit interface — contextless overloads default to empty context.
    override fun predict(sample: Sample): Double = predict(sample, Context.Empty)
    override fun train(sample: Sample, reward: Double, weight: Double) = train(sample, Context.Empty, reward, weight)
    override fun update(sample: Sample, reward: Double, weight: Double) = update(sample, Context.Empty, reward, weight)
    override fun chooseOrThrow(): Sample = chooseOrThrow(Context.Empty)
    override fun optimalOrThrow(): Sample = optimalOrThrow(Context.Empty)

    override fun importData(data: LinearData) {
        val ratio = if (data.step <= 0L) 1f
        else data.step.toFloat() / (linearModel.step + data.step).toFloat()
        linearModel.importData(data, ratio, ratio)
    }

    override fun exportData(): LinearData = linearModel.exportData()
}
