@file:JvmName("Transforms")

package combo.math

import kotlin.jvm.JvmName
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * Bidirectional scalar transform: a link function for GLM-style bandits where the
 * model fits a latent linear predictor and [apply] maps it onto the response scale
 * (and [inverse] back). Used by [combo.bandit.glm.LinearModel] to turn raw weight
 * dot-products into bounded / positive-only / etc. predictions.
 */
interface Transform {
    fun apply(value: Float): Float
    fun inverse(value: Float): Float = throw UnsupportedOperationException("Inverse not available.")

    fun andThen(after: Transform) = object : Transform {
        override fun inverse(value: Float) = this@Transform.inverse(after.inverse(value))
        override fun apply(value: Float) = after.apply(this@Transform.apply(value))
    }
}

object IdentityTransform : Transform {
    override fun inverse(value: Float) = value
    override fun apply(value: Float) = value
}

object LogTransform : Transform {
    override fun inverse(value: Float) = exp(value)
    override fun apply(value: Float) = ln(value)
}

object LogitTransform : Transform {
    override fun apply(value: Float) = 1f / (1f + exp(-value))
    override fun inverse(value: Float) = -ln(1f / value - 1f)
}

object NegativeInverseTransform : Transform {
    override fun inverse(value: Float) = -1f / value
    override fun apply(value: Float) = -1f / value
}

object InverseSquaredTransform : Transform {
    override fun inverse(value: Float) = 1f / sqrt(value)
    override fun apply(value: Float) = 1f / (value * value)
}
