package com.eignex.combo.bo

import com.eignex.koblas.DenseMatrix
import com.eignex.kumulant.stat.summary.MeanStat
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.ln1p
import kotlin.random.Random

/** A fixed acquisition objective scoring an entire set of feature rows together. */
fun interface JointAcquisition {
    /** Set score, always maximized; log acquisitions return negative infinity for zero utility. */
    fun evaluate(features: DenseMatrix): Double
}

/** Frozen model, outcome, and same-context completed/pending coordinates for joint acquisition. */
class JointAcquisitionContext(
    /** Immutable fitted model exposing joint moments. */
    val model: FittedSurrogate,
    /** Fixed improvement baseline in original response units, used by noiseless EI. */
    val incumbent: Double?,
    /** Objective orientation and response policy. */
    val outcome: OutcomeSpec,
    /** Previously evaluated coordinates; noisy EI samples their latent values jointly with candidates. */
    val baseline: DenseMatrix,
    /** Outstanding coordinates whose latent values enter the incremental improvement baseline. */
    val pending: DenseMatrix = DenseMatrix.zero(0, baseline.cols),
)

/** Bind a joint acquisition to a frozen posterior and fixed Monte Carlo random stream. */
fun interface JointAcquisitionFactory {
    /** Randomness is captured here, never redrawn for successive evaluations of the same set. */
    fun create(context: JointAcquisitionContext, random: Random): JointAcquisition
}

/** Normal base-sample construction, frozen within a joint acquisition objective. */
enum class JointNormalSampling {
    /** Seeded pseudo-random draws paired with their negatives; supports any positive even sample count. */
    Antithetic,

    /**
     * Kumulant's LMS-scrambled Sobol normal points. Requires a power-of-two sample count
     * and at most 1,024 latent coordinates, including the baseline and candidates.
     */
    ScrambledSobol,
}

/**
 * Monte Carlo q-EI using a Gaussian approximation to joint latent moments and a fixed incumbent.
 * Pending points raise the sampled improvement threshold; their own improvement is not counted.
 * Log-response models are sampled in Gaussian log coordinates and scored in original units.
 * Reuses fixed normal samples (antithetic by default), canonicalizes row order, and deduplicates coordinates.
 * Near-singular covariance can receive relative numerical jitter up to 1e-8; this is not observation noise.
 * Each bound objective retains common normal samples and a baseline factor, and is single-threaded.
 *
 * @property samples Even number of Monte Carlo draws per evaluated set.
 * @property margin Required improvement in original response units.
 */
class JointExpectedImprovement(val samples: Int = 256, val margin: Double = 0.0) : JointAcquisitionFactory {
    private var selectedSampling = JointNormalSampling.Antithetic

    /** Normal base-sample policy captured when the objective is created. */
    val sampling: JointNormalSampling get() = selectedSampling

    /** Select normal base samples; scrambled Sobol requires a power-of-two draw count. */
    constructor(
        sampling: JointNormalSampling,
        samples: Int = 256,
        margin: Double = 0.0,
    ) : this(samples, margin) {
        validateSampling(samples, sampling)
        selectedSampling = sampling
    }

    init {
        validateJointSettings(samples, margin)
    }
    override fun create(context: JointAcquisitionContext, random: Random): JointAcquisition {
        require(context.incumbent?.isFinite() == true) { "q-EI requires a finite incumbent" }
        return jointImprovement(context, random, samples, margin, noisy = false, sampling = sampling)
    }
}

/**
 * Monte Carlo q-NEI sampling candidates and completed/pending baselines from one joint latent Gaussian.
 * No observed best response is treated as certain. Samples exclude observation noise, and pending labels
 * never enter recorded history. Shares [JointExpectedImprovement]'s sampling and numerical conventions.
 * Baseline prediction work, latent draws, and its Cholesky factor are cached for the bound round.
 * Each candidate set factors only its conditional covariance. Baselines are not pruned.
 *
 * @property samples Even number of Monte Carlo draws per evaluated set.
 * @property margin Required improvement in original response units.
 */
class JointNoisyExpectedImprovement(val samples: Int = 256, val margin: Double = 0.0) : JointAcquisitionFactory {
    private var selectedSampling = JointNormalSampling.Antithetic

    /** Normal base-sample policy captured when the objective is created. */
    val sampling: JointNormalSampling get() = selectedSampling

    /** Select normal base samples; scrambled Sobol requires a power-of-two draw count. */
    constructor(
        sampling: JointNormalSampling,
        samples: Int = 256,
        margin: Double = 0.0,
    ) : this(samples, margin) {
        validateSampling(samples, sampling)
        selectedSampling = sampling
    }

    init {
        validateJointSettings(samples, margin)
    }
    override fun create(context: JointAcquisitionContext, random: Random): JointAcquisition {
        require(context.baseline.rows > 0) { "q-NEI requires completed baseline coordinates" }
        return jointImprovement(context, random, samples, margin, noisy = true, sampling = sampling)
    }
}

private fun validateSampling(samples: Int, sampling: JointNormalSampling) {
    require(sampling != JointNormalSampling.ScrambledSobol || samples and (samples - 1) == 0) {
        "scrambled Sobol requires a power-of-two sample count"
    }
}

private fun validateJointSettings(samples: Int, margin: Double) {
    require(samples > 0 && samples % 2 == 0)
    require(margin >= 0.0 && margin.isFinite())
}

/**
 * Logarithm of smoothed joint EI, using fat-tailed positive-part and batch-maximum approximations.
 * This is not the logarithm of a numerically evaluated EI: negative sampled improvements keep finite,
 * ordered log scores. Temperatures use original response units (positive part) and log utility (batch max).
 * The smoothing follows Ament et al., Unexpected Improvements to Expected Improvement (2023).
 *
 * @property samples Even Monte Carlo draw count.
 * @property margin Required incremental improvement in original response units.
 * @property temperature Positive-part smoothing temperature in original response units.
 * @property batchTemperature Smooth maximum temperature in log-improvement coordinates.
 */
class JointLogExpectedImprovement(
    val samples: Int = 256,
    val margin: Double = 0.0,
    val temperature: Double = 1e-6,
    val batchTemperature: Double = 0.01,
) : JointAcquisitionFactory {
    private var selectedSampling = JointNormalSampling.Antithetic

    /** Normal base-sample policy captured when the objective is created. */
    val sampling: JointNormalSampling get() = selectedSampling

    /** Select normal base samples; scrambled Sobol requires a power-of-two draw count. */
    constructor(
        sampling: JointNormalSampling,
        samples: Int = 256,
        margin: Double = 0.0,
        temperature: Double = 1e-6,
        batchTemperature: Double = 0.01,
    ) : this(samples, margin, temperature, batchTemperature) {
        validateSampling(samples, sampling)
        selectedSampling = sampling
    }

    init {
        validateJointSettings(samples, margin)
        validateTemperatures(temperature, batchTemperature)
    }
    override fun create(context: JointAcquisitionContext, random: Random): JointAcquisition {
        require(context.incumbent?.isFinite() == true) { "q-LogEI requires a finite incumbent" }
        return jointImprovement(
            context,
            random,
            samples,
            margin,
            noisy = false,
            temperature,
            batchTemperature,
            sampling,
        )
    }
}

/**
 * Logarithm of smoothed joint noisy EI, integrating an uncertain latent completed/pending baseline.
 * Shares [JointLogExpectedImprovement]'s approximation and [JointNoisyExpectedImprovement]'s factor cache.
 * Exact baseline duplicates have no incremental utility and return negative infinity when alone.
 *
 * @property samples Even Monte Carlo draw count.
 * @property margin Required incremental improvement in original response units.
 * @property temperature Positive-part smoothing temperature in original response units.
 * @property batchTemperature Smooth maximum temperature in log-improvement coordinates.
 */
class JointLogNoisyExpectedImprovement(
    val samples: Int = 256,
    val margin: Double = 0.0,
    val temperature: Double = 1e-6,
    val batchTemperature: Double = 0.01,
) : JointAcquisitionFactory {
    private var selectedSampling = JointNormalSampling.Antithetic

    /** Normal base-sample policy captured when the objective is created. */
    val sampling: JointNormalSampling get() = selectedSampling

    /** Select normal base samples; scrambled Sobol requires a power-of-two draw count. */
    constructor(
        sampling: JointNormalSampling,
        samples: Int = 256,
        margin: Double = 0.0,
        temperature: Double = 1e-6,
        batchTemperature: Double = 0.01,
    ) : this(samples, margin, temperature, batchTemperature) {
        validateSampling(samples, sampling)
        selectedSampling = sampling
    }

    init {
        validateJointSettings(samples, margin)
        validateTemperatures(temperature, batchTemperature)
    }
    override fun create(context: JointAcquisitionContext, random: Random): JointAcquisition {
        require(context.baseline.rows > 0) { "q-LogNEI requires completed baseline coordinates" }
        return jointImprovement(context, random, samples, margin, noisy = true, temperature, batchTemperature, sampling)
    }
}

private fun validateTemperatures(temperature: Double, batchTemperature: Double) {
    require(temperature > 0.0 && temperature.isFinite() && batchTemperature > 0.0 && batchTemperature.isFinite())
}

private fun jointImprovement(
    context: JointAcquisitionContext,
    random: Random,
    samples: Int,
    margin: Double,
    noisy: Boolean,
    temperature: Double? = null,
    batchTemperature: Double = 0.01,
    sampling: JointNormalSampling = JointNormalSampling.Antithetic,
): JointAcquisition {
    val dimensions = context.baseline.cols
    require(context.pending.cols == dimensions)
    val fixed = ((if (noisy) featureKeys(context.baseline) else emptyList()) + featureKeys(context.pending))
        .distinct().sortedWith(featureOrdering)
    val model = context.model
    val logarithmic = model is LogResponseSurrogate
    val features = featureMatrix(fixed, dimensions)
    requireNotNull(
        if (model is LogResponseSurrogate) {
            model.predictLogJoint(
                DenseMatrix.zero(0, dimensions),
            )
        } else {
            model.predictJoint(DenseMatrix.zero(0, dimensions))
        },
    ) {
        "joint EI requires joint Gaussian moments; marginal-only models are unsupported"
    }
    val posterior = requireNotNull(
        if (model is LogResponseSurrogate) model.logJointPosterior(features) else model.jointPosterior(features),
    ) {
        "joint EI requires joint Gaussian moments; marginal-only models are unsupported"
    }
    val sampler = JointGaussianSampler(posterior, samples, random.nextLong(), sampling)
    fun score(value: Double): Double = context.outcome.score(if (logarithmic) exp(value) else value).also {
        require(it.isFinite()) { "nonfinite joint response; rescale responses" }
    }
    val thresholds = DoubleArray(samples) { draw ->
        var best = if (noisy) Double.NEGATIVE_INFINITY else context.outcome.score(context.incumbent!!)
        for (row in fixed.indices) best = maxOf(best, score(sampler.baselineDraws[row, draw]))
        best
    }
    val fixedKeys = fixed.toSet()
    val values = DoubleArray(samples)
    val smoothing = temperature?.let { LogImprovementSmoothing(it, batchTemperature) }
    var improvements = DoubleArray(0)
    return JointAcquisition { candidates ->
        require(candidates.cols == dimensions)
        val novel = (featureKeys(candidates).distinct() - fixedKeys).sortedWith(featureOrdering)
        if (novel.isEmpty()) return@JointAcquisition if (temperature == null) 0.0 else Double.NEGATIVE_INFINITY
        val draws = sampler.draw(featureMatrix(novel, dimensions))
        if (improvements.size != novel.size) improvements = DoubleArray(novel.size)
        val logBatchSize = ln(novel.size.toDouble())
        for (draw in 0 until samples) {
            for (row in novel.indices) {
                improvements[row] = (score(draws[row, draw]) - thresholds[draw] - margin)
                    .also { require(it.isFinite()) }
            }
            values[draw] = smoothing?.evaluate(improvements, logBatchSize) ?: maxOf(0.0, improvements.max())
        }
        val shift = if (temperature == null) 0.0 else values.max()
        val utility = MeanStat()
        for (value in values) utility.update(if (temperature == null) value else exp(value - shift), 0L)
        val mean = utility.read(0L).mean
        (
            if (temperature == null) {
                mean
            } else {
                shift + ln(
                    mean,
                )
            }
            ).also { require(it.isFinite()) { "nonfinite joint acquisition" } }
    }
}

// Acquisition-specific smoothing and scratch state; the bound objective is already single-threaded.
private class LogImprovementSmoothing(private val temperature: Double, private val batchTemperature: Double) {
    private val logTemperature = ln(temperature)
    private val weights = MeanStat()

    fun evaluate(improvements: DoubleArray, logBatchSize: Double): Double {
        for (i in improvements.indices) {
            val difference = improvements[i]
            val t = difference / temperature
            val logSoftplus = logTemperature + when {
                t > 20.0 -> ln(difference) - logTemperature
                t < -40.0 -> t
                else -> ln(ln1p(exp(t)))
            }
            val ratio = ln(abs(difference)) - logTemperature
            val logDenominator = if (ratio > 0.0) 2.0 * ratio + ln1p(exp(-2.0 * ratio)) else ln1p(exp(2.0 * ratio))
            val tail = ln(0.1) + logTemperature - logDenominator
            val largest = maxOf(logSoftplus, tail)
            improvements[i] = largest + ln1p(exp(minOf(logSoftplus, tail) - largest))
        }
        if (improvements.size == 1) return improvements[0]
        val largest = improvements.max()
        weights.reset()
        for (value in improvements) {
            val gap = (largest - value) / batchTemperature
            val weight = if (gap > 1e150) 0.0 else 2.0 / (2.0 + 2.0 * gap + gap * gap)
            weights.update(weight, 0L)
        }
        return largest + batchTemperature * (ln(weights.read(0L).mean) + logBatchSize)
    }
}
