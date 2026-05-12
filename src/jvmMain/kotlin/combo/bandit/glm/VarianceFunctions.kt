package combo.bandit.glm

import combo.math.*

interface VarianceFunction {
    fun variance(mean: Float): Float
    fun canonicalLink(): Transform
}

object NormalVariance : VarianceFunction {
    override fun variance(mean: Float) = 1.0f
    override fun canonicalLink() = IdentityTransform
}

object BinomialVariance : VarianceFunction {
    override fun variance(mean: Float) = mean * (1 - mean)
    override fun canonicalLink() = LogitTransform
}

typealias BernoulliVariance = BinomialVariance

object PoissonVariance : VarianceFunction {
    override fun variance(mean: Float) = mean
    override fun canonicalLink() = LogTransform
}

object InverseGaussianVariance : VarianceFunction {
    override fun variance(mean: Float) = mean * mean * mean
    override fun canonicalLink() = InverseSquaredTransform
}

object ExponentialVariance : VarianceFunction {
    override fun variance(mean: Float) = 1.0f / (mean * mean)
    override fun canonicalLink() = NegativeInverseTransform
}

/*
TODO
object GammaVariance : VarianceFunction {
    override fun variance(mean: Float) = mean * mean
    override fun canonicalLink() = NegativeInverseTransform
}

object QuasiBinomialVariance : VarianceFunction {
    override fun variance(mean: Float) = mean * (1 - mean)
    override fun canonicalLink() = LogitTransform
}

object QuasiPoissonVariance : VarianceFunction {
    override fun variance(mean: Float) = mean
    override fun canonicalLink() = LogTransform
}
 */
