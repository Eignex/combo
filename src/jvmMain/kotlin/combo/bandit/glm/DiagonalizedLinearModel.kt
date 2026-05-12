package combo.bandit.glm

import combo.math.*
import combo.sat.NumericalInstabilityException
import combo.sat.Problem
import kotlin.math.sqrt
import kotlin.random.Random

class DiagonalizedLinearModel(val family: VarianceFunction,
                              link: Transform,
                              loss: Transform,
                              regularization: Transform,
                              regularizationFactor: Float,
                              val learningRate: LearningRateSchedule,
                              exploration: Float,
                              step: Long,
                              weights: Vector,
                              val precision: Vector,
                              bias: Float,
                              var biasPrecision: Float)
    : LinearModel(link, loss, regularization, regularizationFactor, exploration, step, weights, bias) {

    override fun sample(rng: Random, weights: VectorView) =
            vectors.zeroVector(weights.size).apply {
                transformIndexed { i, _ ->
                    rng.nextNormal(weights[i], sqrt(exploration / precision[i]))
                }
            }

    override fun train(input: VectorView, result: Float, weight: Float) {
        step++
        val yhat = predict(input)
        if (yhat.isNaN() || yhat.isInfinite())
            throw NumericalInstabilityException("Predicted value is NaN or infinite.")
        val diff = yhat - result
        val loss = loss.apply(diff)
        val varF = family.variance(yhat)
        val lr = learningRate.rate(step)
        for (i in input) {
            val reg = regularizationFactor * regularization.apply(weights[i])
            precision[i] += varF * input[i] * input[i]
            val grad = input[i] * (loss + reg) * weight
            weights[i] -= lr * grad / precision[i]
        }
        biasPrecision += varF
        bias -= lr * weight * loss / biasPrecision
    }

    override fun importData(data: LinearData, varianceMixin: Float, weightMixin: Float) {
        require(data.updaterData.size == 1) { "Expected updaterData to have one row." }
        step = (step * (1 - varianceMixin) + data.step * varianceMixin).toLong()
        bias = combineMean(bias, data.bias, 1 - weightMixin, weightMixin)
        biasPrecision = combinePrecision(biasPrecision, data.biasPrecision, bias, data.bias, 1 - varianceMixin, varianceMixin)
        for (i in weights.indices) {
            weights[i] = combineMean(weights[i], data.weights[i], 1 - weightMixin, weightMixin)
            precision[i] = combinePrecision(precision[i], data.updaterData[0][i], weights[i], data.weights[i], 1 - varianceMixin, varianceMixin)
        }
    }

    override fun exportData() = LinearData(weights.toFloatArray(), bias, biasPrecision, step, arrayOf(precision.toFloatArray()))
    override fun blank(variance: Float) = DiagonalizedLinearModel(
            family, link, loss, regularization, regularizationFactor, learningRate, exploration, 0L, vectors.zeroVector(weights.size),
            vectors.zeroVector(precision.size).apply { add(1f / variance) }, bias, 1f / variance)

    class Builder(val size: Int) {

        constructor(problem: Problem) : this(problem.nbrValues)

        private var family: VarianceFunction = NormalVariance
        private var link: Transform? = null
        private var loss: Transform = HuberLoss(0.1f)
        private var exploration: Float = 1f
        private var regularization: Transform = MSELoss
        private var regularizationFactor: Float = 0f
        private var bias: Float? = null
        private var startingStep: Long = 0L
        private var learningRate: LearningRateSchedule = ConstantRate(1f)
        private var priorPrecision: Float = 1f

        /** Starting precision in the diagonals of the variance-covariance matrix. */
        fun priorPrecision(priorPrecision: Float) = apply { this.priorPrecision = priorPrecision }

        /** Additional penalty to how big updates should be. By default 1. */
        fun learningRate(learningRate: LearningRateSchedule) = apply { this.learningRate = learningRate }

        /** Type of response data. For binary data use [BinomialVariance], for normal data use [NormalVariance], etc. */
        fun family(family: VarianceFunction) = apply { this.family = family }

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

        /** Specify GLM link function, otherwise [family]'s [VarianceFunction.canonicalLink] is used. */
        fun link(link: Transform) = apply { this.link = link }

        fun build() =
                DiagonalizedLinearModel(family, link
                        ?: family.canonicalLink(), loss, regularization, regularizationFactor, learningRate,
                        exploration, startingStep, vectors.zeroVector(size),
                        vectors.vector(FloatArray(size) { priorPrecision }),
                        bias ?: if (family is PoissonVariance) 1.01f else 0f,
                        priorPrecision)
    }
}