package combo.bandit

import combo.math.VarianceEstimator
import combo.sat.Instance

data class InstanceData(val instance: Instance, val data: VarianceEstimator) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || other !is InstanceData) return false
        return instance == other.instance && data == other.data
    }

    override fun hashCode(): Int {
        var result = instance.hashCode()
        result = 31 * result + data.hashCode()
        return result
    }
}

class InstancesData(val instances: List<InstanceData>)
    : BanditData, List<InstanceData> by instances {
    override fun migrate(from: IntArray, to: IntArray): BanditData {
        TODO("not implemented")
    }
}