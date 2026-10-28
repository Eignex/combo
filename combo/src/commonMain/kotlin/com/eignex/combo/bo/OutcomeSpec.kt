package com.eignex.combo.bo

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.Vector
import com.eignex.koblas.axpy
import com.eignex.koblas.scale
import com.eignex.kumulant.stat.summary.VarianceStat
import kotlin.math.exp
import kotlin.math.expm1
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/** Which responses define an improvement baseline and observed recommendation. */
enum class ObservationNoise {
    /** Use observed responses as exact function evaluations. */
    Deterministic,

    /** Use weighted replicate means and model means to reduce sensitivity to individual noisy outcomes. */
    Noisy,
}

/** Geometry of the scalar response supplied to a surrogate. */
enum class ResponseScale {
    /** Original signed responses. */
    Identity,

    /**
     * Natural log of positive responses, with lognormal predictions and EI in original units.
     * Known observation variances use a first-order delta approximation in log coordinates.
     */
    Log,
}

/** Objective orientation and optional training-response normalization. */
data class OutcomeSpec(
    /** True for maximization, false for minimization. */
    val maximize: Boolean = true,
    /** Noise policy; weights never stand in for observation noise variances. */
    val noise: ObservationNoise = ObservationNoise.Deterministic,
    /** Fit on standardized responses and map predictions and variance back to reward units. */
    val standardize: Boolean = false,
    /** Response transform applied before optional standardization. */
    val responseScale: ResponseScale = ResponseScale.Identity,
) {
    /** Convert an objective value to a score that is always maximized. */
    fun score(value: Double): Double = if (maximize) value else -value
}

internal class ResponseTransform(val center: Double, val scale: Double, val logarithmic: Boolean = false) {
    fun encode(value: Double): Double {
        require(!logarithmic || value > 0.0) { "log responses must be strictly positive" }
        return ((if (logarithmic) ln(value) else value) - center) / scale
    }
    fun encodeNoise(value: Double, variance: Double): Double {
        if (variance == 0.0) return 0.0
        val deviation = sqrt(variance) / scale / if (logarithmic) value else 1.0
        return deviation * deviation
    }

    private fun decode(values: Vector): Vector = DenseVector.of(values.toDoubleArray()).also {
        it.scale(scale)
        if (center != 0.0) it.axpy(center, DenseVector.of(DoubleArray(it.size) { 1.0 }))
    }

    fun model(fitted: FittedSurrogate): FittedSurrogate {
        fun moments(features: DenseMatrix): PredictiveMoments {
            val prediction = fitted.predict(features)
            return PredictiveMoments(
                decode(prediction.means),
                DenseVector.of(prediction.variances.toDoubleArray()).also { it.scale(scale * scale) },
            )
        }
        fun jointMoments(features: DenseMatrix): JointPredictiveMoments? {
            val joint = fitted.predictJoint(features) ?: return null
            val covariance = DenseMatrix.wrap(
                joint.covariance.rows,
                joint.covariance.cols,
                joint.covariance.values.copyOf(),
            )
            DenseVector.wrap(covariance.values).scale(scale * scale)
            return JointPredictiveMoments(decode(joint.means), covariance)
        }
        fun gaussianJointPosterior(baseline: DenseMatrix): JointPosterior? {
            val posterior = fitted.jointPosterior(baseline) ?: return null
            fun decoded(moments: JointPredictiveMoments): JointPredictiveMoments = JointPredictiveMoments(
                decode(moments.means),
                DenseMatrix.wrap(
                    moments.covariance.rows,
                    moments.covariance.cols,
                    moments.covariance.values.copyOf(),
                ).also { DenseVector.wrap(it.values).scale(scale * scale) },
            )
            return object : JointPosterior {
                override val baseline = decoded(posterior.baseline)
                override fun predict(features: DenseMatrix): JointPosteriorPrediction {
                    val result = posterior.predict(features)
                    return JointPosteriorPrediction(
                        decoded(result.candidates),
                        DenseMatrix.wrap(
                            result.crossCovariance.rows,
                            result.crossCovariance.cols,
                            result.crossCovariance.values.copyOf(),
                        ).also { DenseVector.wrap(it.values).scale(scale * scale) },
                    )
                }
            }
        }
        val raw = object : JointPosteriorSurrogate {
            override val uncertaintyKind get() = fitted.uncertaintyKind
            override fun predict(features: DenseMatrix): PredictiveMoments {
                val prediction = moments(features)
                if (!logarithmic) return prediction
                val means = DenseVector.of(
                    DoubleArray(prediction.means.size) {
                        exp(prediction.means[it] + prediction.variances[it] / 2.0)
                    },
                )
                val variances = DenseVector.of(
                    DoubleArray(prediction.means.size) {
                        exp(2.0 * prediction.means[it] + prediction.variances[it]) * expm1(prediction.variances[it])
                    },
                )
                return PredictiveMoments(means, variances)
            }
            override fun predictJoint(features: DenseMatrix): JointPredictiveMoments? {
                val prediction = jointMoments(features) ?: return null
                if (!logarithmic) return prediction
                val covariance = prediction.covariance
                val means = DoubleArray(prediction.means.size) {
                    exp(prediction.means[it] + covariance[it, it] / 2.0)
                }
                val result = DenseMatrix.zero(means.size)
                for (i in means.indices) {
                    for (j in means.indices) {
                        result[i, j] = means[i] * means[j] * expm1(covariance[i, j])
                    }
                }
                return JointPredictiveMoments(DenseVector.of(means), result)
            }
            override fun jointPosterior(baseline: DenseMatrix): JointPosterior? =
                if (logarithmic) bindJointPosterior(baseline, ::predictJoint) else gaussianJointPosterior(baseline)
            override fun drawFunction(random: Random): SampledFunction {
                val draw = fitted.drawFunction(random)
                return SampledFunction { features ->
                    val values = decode(draw.evaluate(features))
                    if (logarithmic) DenseVector.of(DoubleArray(values.size) { exp(values[it]) }) else values
                }
            }
        }
        val wrapped = if (logarithmic) {
            object : LogJointPosteriorSurrogate, FittedSurrogate by raw {
                override fun jointPosterior(baseline: DenseMatrix) = raw.jointPosterior(baseline)
                override fun predictLog(features: DenseMatrix) = moments(features)
                override fun predictLogJoint(features: DenseMatrix) = jointMoments(features)
                override fun logJointPosterior(baseline: DenseMatrix) = gaussianJointPosterior(baseline)
            }
        } else {
            raw
        }
        if (fitted !is ConditionableSurrogate) return wrapped
        fun conditioned(temporary: TrainingData): FittedSurrogate = model(
            fitted.condition(
                TrainingData(
                    temporary.features,
                    DenseVector.of(DoubleArray(temporary.responses.size) { encode(temporary.responses[it]) }),
                    temporary.weights,
                    temporary.noiseVariances?.let { values ->
                        DenseVector.of(DoubleArray(values.size) { encodeNoise(temporary.responses[it], values[it]) })
                    },
                ),
            ),
        )
        return if (wrapped is LogResponseSurrogate) {
            object : ConditionableSurrogate, LogJointPosteriorSurrogate, FittedSurrogate by wrapped {
                override fun jointPosterior(baseline: DenseMatrix) = wrapped.jointPosterior(baseline)
                override fun predictLog(features: DenseMatrix) = wrapped.predictLog(features)
                override fun predictLogJoint(features: DenseMatrix) = wrapped.predictLogJoint(features)
                override fun logJointPosterior(baseline: DenseMatrix) = wrapped.logJointPosterior(baseline)
                override fun condition(temporary: TrainingData) = conditioned(temporary)
            }
        } else {
            object : ConditionableSurrogate, JointPosteriorSurrogate, FittedSurrogate by wrapped {
                override fun jointPosterior(baseline: DenseMatrix) = wrapped.jointPosterior(baseline)
                override fun condition(temporary: TrainingData) = conditioned(temporary)
            }
        }
    }

    companion object {
        fun from(observations: List<Observation>, outcome: OutcomeSpec): ResponseTransform {
            val logarithmic = outcome.responseScale == ResponseScale.Log
            require(!logarithmic || observations.all { it.reward > 0.0 }) { "log responses must be positive" }
            if (!outcome.standardize || observations.isEmpty()) return ResponseTransform(0.0, 1.0, logarithmic)
            val stat = VarianceStat()
            for (row in observations) stat.update(if (logarithmic) ln(row.reward) else row.reward, 0L, row.weight)
            val result = stat.read(0L)
            return ResponseTransform(
                result.mean,
                sqrt(result.variance).takeIf { it > 0.0 && it.isFinite() } ?: 1.0,
                logarithmic,
            )
        }
    }
}
