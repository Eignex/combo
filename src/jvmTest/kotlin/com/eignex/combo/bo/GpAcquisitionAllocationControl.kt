package com.eignex.combo.bo

import com.eignex.koblas.DenseMatrix
import com.eignex.kumulant.stat.summary.MeanStat
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.ln1p
import kotlin.random.Random

// Frozen pre-optimization control from 53ead84, retained only for paired allocation/timing measurements.
// Keep the old per-draw allocations: changing this control would invalidate before/after comparisons.
internal fun legacyGpImprovement(
    context: JointAcquisitionContext,
    random: Random,
    samples: Int,
    margin: Double,
    noisy: Boolean,
    temperature: Double? = null,
    batchTemperature: Double = 0.01,
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
    val sampler = JointGaussianSampler(posterior, samples, random.nextLong())
    fun score(value: Double): Double = context.outcome.score(if (logarithmic) exp(value) else value).also {
        require(it.isFinite()) { "nonfinite joint response; rescale responses" }
    }
    val thresholds = DoubleArray(samples) { draw ->
        var best = if (noisy) Double.NEGATIVE_INFINITY else context.outcome.score(context.incumbent!!)
        for (row in fixed.indices) best = maxOf(best, score(sampler.baselineDraws[row, draw]))
        best
    }
    return JointAcquisition { candidates ->
        require(candidates.cols == dimensions)
        val novel = (featureKeys(candidates).distinct() - fixed.toSet()).sortedWith(featureOrdering)
        if (novel.isEmpty()) return@JointAcquisition if (temperature == null) 0.0 else Double.NEGATIVE_INFINITY
        val draws = sampler.draw(featureMatrix(novel, dimensions))
        val values = DoubleArray(samples) { draw ->
            val improvements = DoubleArray(novel.size) { row ->
                (score(draws[row, draw]) - thresholds[draw] - margin).also { require(it.isFinite()) }
            }
            if (temperature == null) {
                maxOf(
                    0.0,
                    improvements.max(),
                )
            } else {
                logSmoothedBatchImprovement(improvements, temperature, batchTemperature)
            }
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

// Acquisition-specific smoothing, evaluated directly in log units instead of logging rounded EI.
private fun logSmoothedBatchImprovement(
    improvements: DoubleArray,
    temperature: Double,
    batchTemperature: Double,
): Double {
    val logs = DoubleArray(improvements.size) { i ->
        val difference = improvements[i]
        val t = difference / temperature
        val logSoftplus = ln(temperature) + when {
            t > 20.0 -> ln(difference) - ln(temperature)
            t < -40.0 -> t
            else -> ln(ln1p(exp(t)))
        }
        val ratio = ln(abs(difference)) - ln(temperature)
        val logDenominator = if (ratio > 0.0) 2.0 * ratio + ln1p(exp(-2.0 * ratio)) else ln1p(exp(2.0 * ratio))
        val tail = ln(0.1) + ln(temperature) - logDenominator
        val largest = maxOf(logSoftplus, tail)
        largest + ln1p(exp(minOf(logSoftplus, tail) - largest))
    }
    val largest = logs.max()
    val weights = MeanStat()
    for (value in logs) {
        val gap = (largest - value) / batchTemperature
        val weight = if (gap > 1e150) 0.0 else 2.0 / (2.0 + 2.0 * gap + gap * gap)
        weights.update(weight, 0L)
    }
    return largest + batchTemperature * (ln(weights.read(0L).mean) + ln(logs.size.toDouble()))
}
