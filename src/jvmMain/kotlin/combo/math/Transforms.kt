@file:JvmName("Transforms")

package combo.math

import kotlin.jvm.JvmName
import kotlin.math.*

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

class ShiftTransform(val by: Float) : Transform {
    override fun inverse(value: Float) = value - by
    override fun apply(value: Float) = value + by
}

class ScaleTransform(val by: Float) : Transform {
    override fun inverse(value: Float) = value / by
    override fun apply(value: Float) = value * by
}

object ArcSineTransform : Transform {
    override fun inverse(value: Float) = sin(value).pow(2)
    override fun apply(value: Float) = asin(sqrt(value))
}

object LogTransform : Transform {
    override fun inverse(value: Float) = exp(value)
    override fun apply(value: Float) = ln(value)
}

object SquareRootTransform : Transform {
    override fun inverse(value: Float) = value * value
    override fun apply(value: Float) = sqrt(value)
}

object LogitTransform : Transform {
    override fun apply(value: Float) = 1f / (1f + exp(-value))
    override fun inverse(value: Float) = -ln(1f / value - 1f)
}

object ClogLogTransform : Transform {
    override fun inverse(value: Float) = 1f - exp(-exp(value))
    override fun apply(value: Float) = ln(-ln(1f - value))
}

object InverseTransform : Transform {
    override fun inverse(value: Float) = 1f / value
    override fun apply(value: Float) = 1f / value
}

object NegativeInverseTransform : Transform {
    override fun inverse(value: Float) = -1f / value
    override fun apply(value: Float) = -1f / value
}

object InverseSquaredTransform : Transform {
    override fun inverse(value: Float) = 1f / sqrt(value)
    override fun apply(value: Float) = 1f / (value * value)
}

object RectifierTransform : Transform {
    override fun apply(value: Float) = max(0f, value)
}
