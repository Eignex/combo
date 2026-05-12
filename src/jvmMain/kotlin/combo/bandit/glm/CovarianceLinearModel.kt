package combo.bandit.glm

import combo.math.*
import combo.sat.NumericalInstabilityException
import combo.sat.Problem
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Laplace approximation to the full bayesian linear regression.
 */
class CovarianceLinearModel(val family: VarianceFunction,
                            link: Transform,
                            loss: Transform,
                            regularization: Transform,
                            regularizationFactor: Float,
                            val learningRate: LearningRateSchedule,
                            exploration: Float,
                            step: Long,
                            weights: Vector,
                            val covariance: Matrix,
                            val covarianceL: Matrix,
                            bias: Float,
                            var biasPrecision: Float)
    : LinearModel(link, loss, regularization, regularizationFactor, exploration, step, weights, bias) {

    override fun sample(rng: Random, weights: VectorView): Vector {
        val u = vectors.vector(FloatArray(weights.size) { rng.nextNormal(0f, sqrt(exploration)) })
        val wHat = covarianceL * u
        wHat.add(weights)
        return wHat
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

        // H is a matrix of precision
        // x is the input
        // H^-1 is the covariance matrix
        // g is the gradient wrt to the weights, g2 is the 2nd order gradient
        // Note that g2 = x'*x/var
        // The update rule is:
        // H_t = H_t-1 + g2_t-1
        // w_t = w_t-1 - H^-1_t * g_t-1
        //
        // The matrix H is not stored explicitly, but its inverse can be updated incrementally with the
        // Sherman-Woodbury-Morrison formula, like so:
        // (A + uv^T)^-1 = A^-1 - (A^-1*uv^T*A^-1) / (1 + v^T*A^-1*u)
        // Here, uv^T=g2=x'*x/var <=> u=v=x/sqrt(var), so it simplifies to:
        // (H + g2*g2^T)^-1 = H^-1 - (H^-1*g2*g2^T*H^-1) / (1 + g2^T*H^-1*g2) =
        // (H + g2*g2^T)^-1 = H^-1 - z*z^T, where z = H^-1*x / sqrt(var + x^T*H^-1*x)
        // Also, L, the cholesky decomposition of the covariance matrix is needed to generate sample from the model. It
        // is also updated incrementally.

        // Compute z as above
        val z = let {
            val vec = covariance * input
            val denom = sqrt(varF + (input dot vec))
            if (denom == 0f) return
            vec.divide(denom)
            vec
        }

        // Downdate L, L = chol(H^-1)
        var norm = covarianceL.choleskyDowndate(z)
        if (norm > 1f) {
            // Numerical errors have been accrued, to fix we recalculate cholesky decomposition
            for (i in 0 until covariance.rows)
                covariance[i, i] += 1e-5f

            val L = covariance.cholesky()
            for (i in 0 until L.rows)
                for (j in 0 until L.rows)
                    covarianceL[j, i] = L[i, j] // L is lower triangulate, cholesky is upper

            norm = covarianceL.choleskyDowndate(z)
            while (norm > 1f) {
                // Still failed cholesky downdate due to numerical instability
                // Take smaller step instead
                z.divide(norm + 1e-5f)
                norm = covarianceL.choleskyDowndate(z)
            }
        }

        // Downdate covariance matrix, H^-1 = H^-1 - z*z' (outer product)
        for (i in 0 until z.size)
            for (j in 0 until z.size)
                covariance[i, j] -= z[i] * z[j]

        // w_t = w_t-1 - H^-1_t * g_t-1
        val reg = weights.copy().apply { transform { regularization.apply(it) * regularizationFactor } }
        val grad = (input + reg) * (loss * lr * weight)
        val step = covariance * grad
        weights.subtract(step)

        biasPrecision += varF
        bias -= lr * weight * loss / biasPrecision
    }

    override fun importData(data: LinearData, varianceMixin: Float, weightMixin: Float) {
        require(data.updaterData.size == weights.size * 2) { "Expected updaterData to contain covariance and covarianceL." }
        step = (step * (1 - varianceMixin) + data.step * varianceMixin).toLong()
        bias = combineMean(bias, data.bias, 1 - weightMixin, weightMixin)
        biasPrecision = combinePrecision(biasPrecision, data.biasPrecision, bias, data.bias, 1 - varianceMixin, varianceMixin)
        for (i in weights.indices) {
            weights[i] = combineMean(weights[i], data.weights[i], 1 - weightMixin, weightMixin)
            for (j in i until weights.size) {
                val v = combineMean(covariance[i, j], data.updaterData[i][j], 1 - varianceMixin, varianceMixin)
                covariance[i, j] = v
                covariance[j, i] = v
            }
        }
        val L = covariance.cholesky()
        for (i in 0 until L.rows)
            for (j in 0 until L.rows)
                covarianceL[j, i] = L[i, j] // L is lower triangulate, cholesky is upper
    }

    override fun exportData() = LinearData(weights.toFloatArray(), bias, biasPrecision, step, covariance.toArray() + covarianceL.toArray())

    override fun blank(variance: Float): CovarianceLinearModel {
        val covariance = vectors.zeroMatrix(covariance.rows)
        val covarianceL = vectors.zeroMatrix(covarianceL.rows)
        for (i in 0 until covariance.rows) {
            covariance[i, i] = variance
            covarianceL[i, i] = sqrt(variance)
        }
        return CovarianceLinearModel(
                family, link, loss, regularization, regularizationFactor, learningRate, exploration, 0L, vectors.zeroVector(weights.size),
                covariance, covarianceL, bias, 1f / variance)
    }

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
        private var priorVariance: Float = 1f

        /** Starting variance in the diagonals of the variance-covariance matrix. */
        fun priorVariance(priorVariance: Float) = apply { this.priorVariance = priorVariance }

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

        /** Noise added (or multiplied) to weights during [choose]. */
        fun exploration(exploration: Float) = apply { this.exploration = exploration }

        /** Starting step counter. */
        fun startingStep(step: Long) = apply { this.startingStep = step }

        /** Specify GLM link function, otherwise [family]'s [VarianceFunction.canonicalLink] is used. */
        fun link(link: Transform) = apply { this.link = link }

        fun build(): CovarianceLinearModel {
            val n = size
            val Hinv = vectors.zeroMatrix(n, n)
            val L = vectors.zeroMatrix(n, n)
            for (i in 0 until n) {
                Hinv[i, i] = priorVariance
                L[i, i] = sqrt(priorVariance)
            }
            val defaultBias = if (family is PoissonVariance) 1.01f else 0f
            return CovarianceLinearModel(family, link ?: family.canonicalLink(), loss, regularization,
                    regularizationFactor, learningRate, exploration, startingStep, vectors.zeroVector(size), Hinv, L,
                    bias ?: defaultBias, 1f / priorVariance)
        }
    }
}