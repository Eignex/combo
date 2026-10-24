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
        val raw = object : FittedSurrogate {
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
            override fun drawFunction(random: Random): SampledFunction {
                val draw = fitted.drawFunction(random)
                return SampledFunction { features ->
                    val values = decode(draw.evaluate(features))
                    if (logarithmic) DenseVector.of(DoubleArray(values.size) { exp(values[it]) }) else values
                }
            }
        }
        val wrapped = if (logarithmic) {
            object : LogResponseSurrogate, FittedSurrogate by raw {
                override fun predictLog(features: DenseMatrix) = moments(features)
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
            object : ConditionableSurrogate, LogResponseSurrogate, FittedSurrogate by wrapped {
                override fun predictLog(features: DenseMatrix) = wrapped.predictLog(features)
                override fun condition(temporary: TrainingData) = conditioned(temporary)
            }
        } else {
            object : ConditionableSurrogate, FittedSurrogate by wrapped {
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
