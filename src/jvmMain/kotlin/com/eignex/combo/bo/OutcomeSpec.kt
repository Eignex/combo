package com.eignex.combo.bo

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.Vector
import com.eignex.koblas.axpy
import com.eignex.koblas.scale
import com.eignex.kumulant.stat.summary.VarianceStat
import kotlin.math.sqrt
import kotlin.random.Random

/** Which responses define an improvement baseline and observed recommendation. */
enum class ObservationNoise {
    /** Use observed responses as exact function evaluations. */
    Deterministic,

    /** Use weighted replicate means and model means to reduce sensitivity to individual noisy outcomes. */
    Noisy,
}

/** Objective orientation and optional training-response normalization. */
data class OutcomeSpec(
    /** True for maximization, false for minimization. */
    val maximize: Boolean = true,
    /** Noise policy; weights never stand in for observation noise variances. */
    val noise: ObservationNoise = ObservationNoise.Deterministic,
    /** Fit on standardized responses and map predictions and variance back to reward units. */
    val standardize: Boolean = false,
) {
    /** Convert an objective value to a score that is always maximized. */
    fun score(value: Double): Double = if (maximize) value else -value
}

internal class ResponseTransform(val center: Double, val scale: Double) {
    fun encode(value: Double): Double = (value - center) / scale
    fun decode(value: Double): Double = center + scale * value

    private fun decode(values: Vector): Vector = DenseVector.of(values.toDoubleArray()).also {
        it.scale(scale)
        if (center != 0.0) it.axpy(center, DenseVector.of(DoubleArray(it.size) { 1.0 }))
    }

    fun model(fitted: FittedSurrogate): FittedSurrogate {
        val wrapped = object : FittedSurrogate {
            override val uncertaintyKind get() = fitted.uncertaintyKind
            override fun predict(features: DenseMatrix): PredictiveMoments {
                val moments = fitted.predict(features)
                return PredictiveMoments(
                    decode(moments.means),
                    DenseVector.of(moments.variances.toDoubleArray()).also { it.scale(scale * scale) },
                )
            }
            override fun drawFunction(random: Random): SampledFunction {
                val draw = fitted.drawFunction(random)
                return SampledFunction { features ->
                    val values = draw.evaluate(features)
                    decode(values)
                }
            }
        }
        if (fitted !is ConditionableSurrogate) return wrapped
        return object : ConditionableSurrogate, FittedSurrogate by wrapped {
            override fun condition(temporary: TrainingData): FittedSurrogate = model(
                fitted.condition(
                    TrainingData(
                        temporary.features,
                        DenseVector.of(DoubleArray(temporary.responses.size) { encode(temporary.responses[it]) }),
                        temporary.weights,
                        temporary.noiseVariances?.let { values ->
                            DenseVector.of(DoubleArray(values.size) { values[it] / (scale * scale) })
                        },
                    ),
                ),
            )
        }
    }

    companion object {
        fun from(observations: List<Observation>, enabled: Boolean): ResponseTransform {
            if (!enabled || observations.isEmpty()) return ResponseTransform(0.0, 1.0)
            val stat = VarianceStat()
            for (row in observations) stat.update(row.reward, 0L, row.weight)
            val result = stat.read(0L)
            return ResponseTransform(result.mean, sqrt(result.variance).takeIf { it > 0.0 && it.isFinite() } ?: 1.0)
        }
    }
}
