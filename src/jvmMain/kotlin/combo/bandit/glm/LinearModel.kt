package combo.bandit.glm

import combo.math.Transform
import combo.math.Vector
import combo.math.VectorView
import kotlin.random.Random

abstract class LinearModel(val link: Transform,
                           val loss: Transform,
                           val regularization: Transform,
                           val regularizationFactor: Float,
                           val exploration: Float,
                           var step: Long,
                           val weights: Vector,
                           var bias: Float) {

    fun predict(input: VectorView) = link.apply(bias + (input dot weights))
    abstract fun sample(rng: Random, weights: VectorView = this.weights): VectorView

    abstract fun train(input: VectorView, result: Float, weight: Float)
    open fun trainAll(inputs: Array<out VectorView>, results: FloatArray, weights: FloatArray?) {
        for (i in inputs.indices)
            train(inputs[i], results[i], weights?.get(i) ?: 1.0f)
    }

    /** Create a reset copy of the model with the given variance parameter and cleared weights. */
    abstract fun blank(variance: Float): LinearModel

    /**
     * @param varianceMixin ratio between 0-1 of how much data should change variance estimate
     * @param weightMixin ratio between 0-1 of how much data should change weights
     */
    abstract fun importData(data: LinearData, varianceMixin: Float = 1f, weightMixin: Float = 1f)

    abstract fun exportData(): LinearData
}
