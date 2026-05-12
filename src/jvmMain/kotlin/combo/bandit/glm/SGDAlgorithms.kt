package combo.bandit.glm

import combo.math.EMPTY_MATRIX
import combo.math.Matrix
import combo.math.Vector
import combo.math.vectors
import kotlin.math.exp
import kotlin.math.sqrt

interface SGDAlgorithm {
    fun step(weight: Float, i: Int, grad: Float, t: Long): Float
    fun importData(data: Matrix, varianceMixin: Float, weightMixin: Float)
    fun exportData(): Matrix
    fun copyReset(): SGDAlgorithm
}

class SGD(val learningRate: LearningRateSchedule) : SGDAlgorithm {

    override fun step(weight: Float, i: Int, grad: Float, t: Long): Float {
        val eta = learningRate.rate(t)
        return weight - eta * grad
    }

    override fun importData(data: Matrix, varianceMixin: Float, weightMixin: Float) {}
    override fun exportData() = EMPTY_MATRIX
    override fun copyReset() = this
}

interface LearningRateSchedule {
    fun rate(t: Long): Float
}

class ConstantRate(val eta: Float = 0.001f) : LearningRateSchedule {
    override fun rate(t: Long) = eta
}

class StepDecay(val eta: Float = 0.01f, val k: Float = 1e-3f) : LearningRateSchedule {
    override fun rate(t: Long) = eta / (1 + k * t)
}

class ExponentialDecay(val eta: Float = 0.01f, val k: Float = 1e-5f) : LearningRateSchedule {
    override fun rate(t: Long) = eta * exp(-k * t)
}

class AdaGrad(val v: Vector, val eta: Float = 0.01f, val eps: Float = 1e-5f) : SGDAlgorithm {
    constructor(n: Int, eta: Float = 0.01f, eps: Float = 1e-5f) : this(vectors.zeroVector(n), eta, eps)

    override fun step(weight: Float, i: Int, grad: Float, t: Long): Float {
        v[i] += grad * grad
        return weight - grad * eta / sqrt(v[i] + eps)
    }

    override fun importData(data: Matrix, varianceMixin: Float, weightMixin: Float) {
        v.assign(v * (1 - varianceMixin) + data[0] * varianceMixin)
    }

    override fun exportData() = vectors.matrix(arrayOf(v.toFloatArray()))
    override fun copyReset() = AdaGrad(v.size, eta, eps)
}

class RMSProp(val v: Vector, val eta: Float = 0.01f, val beta: Float = 0.9f, val eps: Float = 1e-5f) : SGDAlgorithm {
    constructor(n: Int, eta: Float = 0.01f, eps: Float = 1e-5f, beta: Float = 0.9f) : this(vectors.zeroVector(n), eta, beta, eps)

    override fun step(weight: Float, i: Int, grad: Float, t: Long): Float {
        v[i] += beta * v[i] + (1 - beta) * grad * grad
        return weight - grad * eta / sqrt(v[i] + eps)
    }

    override fun importData(data: Matrix, varianceMixin: Float, weightMixin: Float) {
        v.assign(v * (1 - varianceMixin) + data[0] * varianceMixin)
    }

    override fun exportData() = vectors.matrix(arrayOf(v.toFloatArray()))
    override fun copyReset() = RMSProp(v.size, eta, eps, beta)
}

class Adam(val m: Vector, val v: Vector, val eta: Float = 0.01f,
           val beta1: Float = 0.9f, val beta2: Float = 0.999f, val eps: Float = 1e-5f) : SGDAlgorithm {
    constructor(n: Int, eta: Float = 0.01f, beta1: Float = 0.9f, beta2: Float = 0.999f, eps: Float = 1e-5f) : this(
            vectors.zeroVector(n), vectors.zeroVector(n), eta, beta1, beta2, eps)

    override fun step(weight: Float, i: Int, grad: Float, t: Long): Float {
        m[i] = beta1 * m[i] + (1 - beta1) * grad
        v[i] = beta2 * v[i] + (1 - beta2) * grad * grad
        return weight - grad * eta * m[i] / sqrt(v[i] + eps)
    }

    override fun importData(data: Matrix, varianceMixin: Float, weightMixin: Float) {
        m.assign(m * (1 - varianceMixin) + data[0] * varianceMixin)
        v.assign(v * (1 - varianceMixin) + data[1] * varianceMixin)
    }

    override fun exportData() = vectors.matrix(arrayOf(m.toFloatArray(), v.toFloatArray()))
    override fun copyReset() = Adam(m.size, eta, beta1, beta2, eps)
}
