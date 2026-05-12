package combo.bandit

import combo.sat.Instance
import kotlin.math.min

interface UpdateEvent {
    fun collectTo(offset: Int, instances: Array<Instance>, results: FloatArray, weights: FloatArray): UpdateEvent?
    val size: Int
}

class BatchUpdate(val instances: Array<Instance>, val results: FloatArray, val weights: FloatArray?) : UpdateEvent {
    override fun collectTo(offset: Int, instances: Array<Instance>, results: FloatArray, weights: FloatArray): UpdateEvent? {
        val n = this.instances.size
        val m = min(this.instances.size, instances.size - offset)
        this.instances.copyInto(instances, offset, 0, m)
        this.results.copyInto(results, offset, 0, m)
        if (this.weights != null)
            this.weights.copyInto(weights, offset, 0, m)
        else
            for (i in 0 until m)
                weights[i + offset] = 1.0f
        return if (n > m) {
            BatchUpdate(this.instances.copyOfRange(m, n),
                    this.results.copyOfRange(m, n),
                    this.weights?.copyOfRange(m, n))
        } else null
    }

    override val size: Int = instances.size


}

class SingleUpdate(val instance: Instance, val result: Float, val weight: Float) : UpdateEvent {
    override fun collectTo(offset: Int, instances: Array<Instance>, results: FloatArray, weights: FloatArray): UpdateEvent? {
        instances[offset] = instance
        results[offset] = result
        weights[offset] = weight
        return null
    }

    override val size: Int get() = 1
}
