@file:JvmName("LossFunctions")

package combo.bandit.glm

import combo.math.IdentityTransform
import combo.math.Transform
import kotlin.jvm.JvmName
import kotlin.math.absoluteValue
import kotlin.math.sign

class HuberLoss(val delta: Float) : Transform {
    override fun apply(value: Float): Float {
        val absoluteValue = value.absoluteValue
        return if (absoluteValue <= delta) value
        else (value * delta) / absoluteValue
    }
}

/**
 * For use with l2 regularized regression
 */
typealias MSELoss = IdentityTransform

/**
 * For use with l1 regularized regression
 */
object AbsoluteLoss : Transform {
    override fun apply(value: Float) = value.sign
}

