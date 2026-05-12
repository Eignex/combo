package combo.bandit.univariate

import com.eignex.kumulant.core.Result
import com.eignex.kumulant.core.SeriesStat
import com.eignex.kumulant.stat.summary.BernoulliSumResult
import com.eignex.kumulant.stat.summary.MomentsResult
import com.eignex.kumulant.stat.summary.WeightedMeanResult
import com.eignex.kumulant.stat.summary.WeightedVarianceResult
import kotlin.math.ln
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Decides which arm to play given snapshots of each arm's sufficient statistic [R].
 * Owns the per-arm stat lifecycle: [createArm] returns a freshly-seeded accumulator,
 * [update] folds a single observation in.
 *
 * Some policies also accumulate global state (e.g. total samples for UCB), exposed
 * via [addArm]/[removeArm] hooks fired by the bandit when arms come and go.
 */
interface BanditPolicy<R : Result> {
    fun createArm(): SeriesStat<R>
    fun update(arm: SeriesStat<R>, value: Double, weight: Double = 1.0) {
        arm.update(value, 0L, weight)
    }
    fun evaluate(snapshot: R, step: Long, maximize: Boolean, rng: Random): Double

    fun addArm(snapshot: R) {}
    fun removeArm(snapshot: R) {}
}

private fun signedMean(mean: Double, maximize: Boolean) = if (maximize) mean else -mean

class ThompsonSampling<R : Result>(
    val posterior: UnivariatePosterior<R>,
) : BanditPolicy<R> {
    override fun createArm() = posterior.createArm()
    override fun update(arm: SeriesStat<R>, value: Double, weight: Double) =
        posterior.update(arm, value, weight)
    override fun evaluate(snapshot: R, step: Long, maximize: Boolean, rng: Random) =
        signedMean(posterior.sample(snapshot, rng), maximize)
}

class UCB1(
    val alpha: Double = 1.0,
    private val priorAlpha: Double = 1.0,
    private val priorBeta: Double = 1.0,
) : BanditPolicy<BernoulliSumResult> {
    private var totalSamples: Double = 0.0

    override fun createArm() = BernoulliSumStatSeeded(priorAlpha, priorBeta)
    override fun update(arm: SeriesStat<BernoulliSumResult>, value: Double, weight: Double) {
        super.update(arm, value, weight)
        totalSamples += weight
    }
    override fun evaluate(snapshot: BernoulliSumResult, step: Long, maximize: Boolean, rng: Random): Double {
        val n = snapshot.trials
        if (n < 1.0) return Double.POSITIVE_INFINITY
        val mean = snapshot.successes / n
        val score = signedMean(mean, maximize)
        return score + alpha * sqrt(2 * ln(totalSamples) / n)
    }
    override fun addArm(snapshot: BernoulliSumResult) { totalSamples += snapshot.trials }
    override fun removeArm(snapshot: BernoulliSumResult) { totalSamples -= snapshot.trials }
}

private fun BernoulliSumStatSeeded(alpha: Double, beta: Double) =
    com.eignex.kumulant.stat.summary.BernoulliSumStat().also {
        if (alpha > 0.0) it.update(1.0, 0L, alpha)
        if (beta > 0.0) it.update(0.0, 0L, beta)
    }

class UCB1Normal(
    val alpha: Double = 1.0,
    private val priorMean: Double = 0.0,
    private val priorWeight: Double = 0.02,
) : BanditPolicy<MomentsResult> {
    private var nbrArms = 0

    override fun createArm() = momentsArm(priorMean, priorWeight)
    override fun evaluate(snapshot: MomentsResult, step: Long, maximize: Boolean, rng: Random): Double {
        val nj = snapshot.totalWeights
        if (nbrArms <= 1 || nj < 8 * ln(nbrArms.toDouble())) return Double.POSITIVE_INFINITY
        val score = signedMean(snapshot.mean, maximize)
        val mos = snapshot.meanOfSquares()
        val p1 = (mos - nj * snapshot.mean * snapshot.mean) / (nj - 1)
        return score + alpha * sqrt(16 * p1 * (ln(nbrArms - 1.0) / nj))
    }
    override fun addArm(snapshot: MomentsResult) { nbrArms++ }
    override fun removeArm(snapshot: MomentsResult) { nbrArms-- }
}

class UCB1Tuned(
    val alpha: Double = 1.0,
    private val priorMean: Double = 0.0,
    private val priorWeight: Double = 0.02,
) : BanditPolicy<MomentsResult> {
    private var totalSamples: Double = 0.0

    override fun createArm() = momentsArm(priorMean, priorWeight)
    override fun update(arm: SeriesStat<MomentsResult>, value: Double, weight: Double) {
        super.update(arm, value, weight)
        totalSamples += weight
    }
    override fun evaluate(snapshot: MomentsResult, step: Long, maximize: Boolean, rng: Random): Double {
        val nj = snapshot.totalWeights
        if (nj <= 1.0) return Double.POSITIVE_INFINITY
        val padding = ln(totalSamples) / nj
        val v = snapshot.meanOfSquares() - snapshot.mean * snapshot.mean + sqrt(2.0 * padding)
        val score = signedMean(snapshot.mean, maximize)
        return score + alpha * sqrt(padding * min(0.25, v))
    }
    override fun addArm(snapshot: MomentsResult) { totalSamples += snapshot.totalWeights }
    override fun removeArm(snapshot: MomentsResult) { totalSamples -= snapshot.totalWeights }
}

class Greedy(
    private val priorMean: Double = 0.0,
    private val priorWeight: Double = 0.02,
    private val priorSquaredDeviations: Double = 0.02,
) : BanditPolicy<WeightedVarianceResult> {
    override fun createArm() = NormalPosterior(priorMean, priorWeight, priorSquaredDeviations).createArm()
    override fun evaluate(snapshot: WeightedVarianceResult, step: Long, maximize: Boolean, rng: Random) =
        signedMean(snapshot.mean, maximize)
}

class EpsilonGreedy(
    val epsilon: Double = 0.1,
    private val priorMean: Double = 0.0,
    private val priorWeight: Double = 0.02,
    private val priorSquaredDeviations: Double = 0.02,
) : BanditPolicy<WeightedVarianceResult> {
    init { require(epsilon in 0.0..1.0) { "epsilon must be in 0..1, got $epsilon" } }

    override fun createArm() = NormalPosterior(priorMean, priorWeight, priorSquaredDeviations).createArm()
    override fun evaluate(snapshot: WeightedVarianceResult, step: Long, maximize: Boolean, rng: Random): Double {
        return if (Random(step).nextDouble() < epsilon) rng.nextDouble()
        else signedMean(snapshot.mean, maximize)
    }
}

class EpsilonDecreasing(
    val epsilon: Double = 2.0,
    val decay: Double = 0.5,
    private val priorMean: Double = 0.0,
    private val priorWeight: Double = 0.02,
    private val priorSquaredDeviations: Double = 0.02,
) : BanditPolicy<WeightedVarianceResult> {
    private var totalSamples: Double = 0.0
    init { require(epsilon > 0.0) { "epsilon must be positive, got $epsilon" } }

    override fun createArm() = NormalPosterior(priorMean, priorWeight, priorSquaredDeviations).createArm()
    override fun update(arm: SeriesStat<WeightedVarianceResult>, value: Double, weight: Double) {
        super.update(arm, value, weight)
        totalSamples += weight
    }
    override fun evaluate(snapshot: WeightedVarianceResult, step: Long, maximize: Boolean, rng: Random): Double {
        val eps = min(1.0, epsilon / totalSamples.pow(decay))
        return if (Random(step).nextDouble() < eps) rng.nextDouble()
        else signedMean(snapshot.mean, maximize)
    }
    override fun addArm(snapshot: WeightedVarianceResult) { totalSamples += snapshot.totalWeights }
    override fun removeArm(snapshot: WeightedVarianceResult) { totalSamples -= snapshot.totalWeights }
}

class UniformSelection(
    private val priorMean: Double = 0.0,
    private val priorWeight: Double = 0.02,
    private val priorSquaredDeviations: Double = 0.02,
) : BanditPolicy<WeightedVarianceResult> {
    override fun createArm() = NormalPosterior(priorMean, priorWeight, priorSquaredDeviations).createArm()
    override fun evaluate(snapshot: WeightedVarianceResult, step: Long, maximize: Boolean, rng: Random) =
        rng.nextDouble()
}
