package combo.bandit.univariate

import com.eignex.kumulant.core.Result
import com.eignex.kumulant.core.SeriesStat
import com.eignex.kumulant.stat.summary.BernoulliSumResult
import com.eignex.kumulant.stat.summary.BernoulliSumStat
import com.eignex.kumulant.stat.summary.MeanStat
import com.eignex.kumulant.stat.summary.MomentsResult
import com.eignex.kumulant.stat.summary.MomentsStat
import com.eignex.kumulant.stat.summary.VarianceStat
import com.eignex.kumulant.stat.summary.WeightedMeanResult
import com.eignex.kumulant.stat.summary.WeightedVarianceResult
import combo.bandit.util.nextBeta
import combo.bandit.util.nextGamma
import combo.bandit.util.nextNormal
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Conjugate posterior over a univariate likelihood, parameterized by the per-arm sufficient
 * statistic [R] (a kumulant [Result]). The posterior owns the per-arm accumulator: it
 * constructs a freshly-seeded one via [createArm], folds observations through [update]
 * (which may transform the value, e.g. log-normal), and draws posterior samples via
 * [sample] given a snapshot.
 */
interface UnivariatePosterior<R : Result> {
    fun createArm(): SeriesStat<R>
    fun update(arm: SeriesStat<R>, value: Double, weight: Double = 1.0) {
        arm.update(value, 0L, weight)
    }
    fun sample(snapshot: R, rng: Random): Double
}

private fun varianceStatSeeded(priorMean: Double, priorWeight: Double): VarianceStat {
    val s = VarianceStat()
    if (priorWeight > 0.0) s.update(priorMean, 0L, priorWeight)
    return s
}

private fun meanStatSeeded(priorMean: Double, priorWeight: Double): MeanStat {
    val s = MeanStat()
    if (priorWeight > 0.0) s.update(priorMean, 0L, priorWeight)
    return s
}

private fun bernoulliSeeded(alpha: Double, beta: Double): BernoulliSumStat {
    val s = BernoulliSumStat()
    if (alpha > 0.0) s.update(1.0, 0L, alpha)
    if (beta > 0.0) s.update(0.0, 0L, beta)
    return s
}

private fun momentsSeeded(priorMean: Double, priorWeight: Double): MomentsStat {
    val s = MomentsStat()
    if (priorWeight > 0.0) s.update(priorMean, 0L, priorWeight)
    return s
}

class BinomialPosterior(
    private val priorAlpha: Double = 1.0,
    private val priorBeta: Double = 1.0,
) : UnivariatePosterior<BernoulliSumResult> {
    override fun createArm() = bernoulliSeeded(priorAlpha, priorBeta)
    override fun sample(snapshot: BernoulliSumResult, rng: Random): Double {
        val alpha = snapshot.successes
        val beta = snapshot.trials - snapshot.successes
        return rng.nextBeta(alpha, beta)
    }
}

class PoissonPosterior(
    private val priorMean: Double = 1.0,
    private val priorWeight: Double = 0.01,
) : UnivariatePosterior<WeightedMeanResult> {
    override fun createArm() = meanStatSeeded(priorMean, priorWeight)
    override fun sample(snapshot: WeightedMeanResult, rng: Random): Double {
        val sum = snapshot.mean * snapshot.totalWeights
        return rng.nextGamma(sum) / snapshot.totalWeights
    }
}

class GeometricPosterior(
    private val priorMean: Double = 2.0,
    private val priorWeight: Double = 1.0,
) : UnivariatePosterior<WeightedMeanResult> {
    override fun createArm() = meanStatSeeded(priorMean, priorWeight)
    override fun sample(snapshot: WeightedMeanResult, rng: Random): Double {
        val sum = snapshot.mean * snapshot.totalWeights
        return rng.nextBeta(snapshot.totalWeights, sum - snapshot.totalWeights)
    }
}

class NormalPosterior(
    private val priorMean: Double = 0.0,
    private val priorWeight: Double = 0.02,
    private val priorSquaredDeviations: Double = 0.02,
) : UnivariatePosterior<WeightedVarianceResult> {
    override fun createArm(): SeriesStat<WeightedVarianceResult> {
        val s = VarianceStat()
        // Seed mean with priorMean at priorWeight; bump SST so variance is non-zero up front.
        if (priorWeight > 0.0) {
            s.update(priorMean, 0L, priorWeight)
            // Inject squared-deviation pseudo-mass via two opposing observations.
            if (priorSquaredDeviations > 0.0) {
                val sigma = sqrt(priorSquaredDeviations / priorWeight)
                s.update(priorMean + sigma, 0L, priorWeight / 2.0)
                s.update(priorMean - sigma, 0L, priorWeight / 2.0)
            }
        }
        return s
    }

    override fun sample(snapshot: WeightedVarianceResult, rng: Random): Double {
        while (true) {
            val n = snapshot.totalWeights
            if (n <= 0.0) return rng.nextNormal(priorMean, sqrt(priorSquaredDeviations))
            val alpha = n / 2.0
            val beta = snapshot.variance * n / 2.0
            val sampleVariance = beta / rng.nextGamma(alpha)
            if (!sampleVariance.isFinite()) continue
            val value = rng.nextNormal(snapshot.mean, sqrt(sampleVariance / n))
            if (value.isFinite()) return value
        }
    }
}

class LogNormalPosterior(
    private val priorMean: Double = 0.0,
    private val priorWeight: Double = 0.02,
    private val priorSquaredDeviations: Double = 2.0,
) : UnivariatePosterior<WeightedVarianceResult> {
    private val backing = NormalPosterior(priorMean, priorWeight, priorSquaredDeviations)

    override fun createArm() = backing.createArm()

    override fun update(arm: SeriesStat<WeightedVarianceResult>, value: Double, weight: Double) {
        arm.update(ln(value), 0L, weight)
    }

    override fun sample(snapshot: WeightedVarianceResult, rng: Random): Double {
        while (true) {
            val n = snapshot.totalWeights
            if (n <= 0.0) return exp(rng.nextNormal(priorMean, sqrt(priorSquaredDeviations)))
            val alpha = n / 2.0
            val beta = snapshot.variance * n / 2.0
            val sampleVariance = beta / rng.nextGamma(alpha)
            if (!sampleVariance.isFinite()) continue
            val sampleMean = rng.nextNormal(snapshot.mean, sqrt(sampleVariance / n))
            val value = exp(sampleMean + sampleVariance / 2.0)
            if (value.isFinite()) return value
        }
    }
}

class ExponentialPosterior(
    private val priorMean: Double = 1.0,
    private val priorWeight: Double = 0.01,
) : UnivariatePosterior<WeightedMeanResult> {
    override fun createArm() = meanStatSeeded(priorMean, priorWeight)
    override fun sample(snapshot: WeightedMeanResult, rng: Random): Double {
        val sum = snapshot.mean * snapshot.totalWeights
        return rng.nextGamma(snapshot.totalWeights) / sum
    }
}

class GammaScalePosterior(
    val fixedShape: Double,
    private val priorMean: Double = 1.0,
    private val priorWeight: Double = 0.1,
) : UnivariatePosterior<WeightedMeanResult> {
    override fun createArm() = meanStatSeeded(priorMean, priorWeight)
    override fun sample(snapshot: WeightedMeanResult, rng: Random): Double {
        val sum = snapshot.mean * snapshot.totalWeights
        return rng.nextGamma(snapshot.totalWeights * fixedShape) / sum
    }
}

/** UCB1Normal and UCB1Tuned need access to second moments; expose them as a posterior-free moments arm. */
internal fun momentsArm(priorMean: Double = 0.0, priorWeight: Double = 0.02): MomentsStat =
    momentsSeeded(priorMean, priorWeight)

/** Helper to read `meanOfSquares` from a moments snapshot (= m2/N + mean^2). */
internal fun MomentsResult.meanOfSquares(): Double =
    if (totalWeights > 0.0) m2 / totalWeights + mean * mean else 0.0
