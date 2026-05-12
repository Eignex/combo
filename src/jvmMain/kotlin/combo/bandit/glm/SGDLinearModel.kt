package combo.bandit.glm

import combo.math.*
import combo.sat.NumericalInstabilityException
import combo.sat.Problem
import kotlin.math.sqrt
import kotlin.random.Random

class SGDLinearModel(link: Transform,
                     loss: Transform,
                     regularization: Transform,
                     regularizationFactor: Float,
                     val updater: SGDAlgorithm,
                     exploration: Float,
                     step: Long,
                     weights: Vector,
                     val biasRate: LearningRateSchedule = ExponentialDecay(),
                     bias: Float)
    : LinearModel(link, loss, regularization, regularizationFactor, exploration, step, weights, bias) {

    override fun sample(rng: Random, weights: VectorView): VectorView {
        return if (exploration > 0f)
            vectors.zeroVector(weights.size).apply {
                transformIndexed { i, _ ->
                    rng.nextNormal(weights[i], sqrt(exploration))
                }
            }
        else weights
    }

    override fun train(input: VectorView, result: Float, weight: Float) {
        step++
        val yhat = predict(input)
        if (yhat.isNaN() || yhat.isInfinite())
            throw NumericalInstabilityException("Predicted value is NaN or infinite.")
        val diff = yhat - result
        val loss = loss.apply(diff)
        for (i in input) {
            val reg = regularizationFactor * regularization.apply(weights[i])
            val grad = input[i] * (loss + reg) * weight
            weights[i] = updater.step(weights[i], i, grad, step)
        }
        bias -= diff * biasRate.rate(step)
    }

    override fun importData(data: LinearData, varianceMixin: Float, weightMixin: Float) {
        step = (step * (1 - varianceMixin) + data.step * varianceMixin).toLong()
        bias = combineMean(bias, data.bias, 1 - weightMixin, weightMixin)
        for (i in weights.indices)
            weights[i] = combineMean(weights[i], data.weights[i], 1 - weightMixin, weightMixin)
        updater.importData(vectors.matrix(data.updaterData), varianceMixin, weightMixin)
    }

    override fun exportData() = LinearData(weights.toFloatArray(), bias, 0f, step, updater.exportData().toArray())
    override fun blank(variance: Float) = SGDLinearModel(
            link, loss, regularization, regularizationFactor, updater.copyReset(), exploration, 0L, vectors.zeroVector(weights.size), biasRate, bias)

    class Builder(val size: Int) {

        constructor(problem: Problem) : this(problem.nbrValues)

        private var link: Transform = IdentityTransform
        private var loss: Transform = HuberLoss(0.1f)
        private var exploration: Float = 0.1f
        private var regularization: Transform = MSELoss
        private var regularizationFactor: Float = 1e-5f
        private var bias: Float? = null
        private var startingStep: Long = 0L
        private var biasRate: LearningRateSchedule = ExponentialDecay()
        private var updater: SGDAlgorithm = SGD(ExponentialDecay())

        /** Initial bias term. This has no effect if importing data. */
        fun bias(bias: Float) = apply { this.bias = bias }

        /** Loss function of the regression, by default it is mean squared loss. */
        fun loss(loss: Transform) = apply { this.loss = loss }

        /** Type of regularization to apply, by default L2. */
        fun regularization(regularization: Transform) = apply { this.regularization = regularization }

        /** Regularization constant. */
        fun regularizationFactor(regularizationFactor: Float) = apply { this.regularizationFactor = regularizationFactor }

        /** Noise added (or multiplied) to weights during choose. */
        fun exploration(exploration: Float) = apply { this.exploration = exploration }

        /** Starting step counter. */
        fun startingStep(step: Long) = apply { this.startingStep = step }

        /** Specify GLM link function. */
        fun link(link: Transform) = apply { this.link = link }

        fun biasRate(biasRate: LearningRateSchedule) = apply { this.biasRate = biasRate }

        /** Stochastic gradient descent algorithm to use. */
        fun updater(updater: SGDAlgorithm) = apply { this.updater = updater }

        fun build() = SGDLinearModel(link, loss, regularization, regularizationFactor, updater, exploration,
                startingStep, vectors.zeroVector(size), biasRate,
                bias ?: if (link is LogTransform) 1.01f else 0f)
    }
}