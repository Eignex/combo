package combo.bandit.glm

import combo.bandit.BanditData

class LinearData(val weights: FloatArray, val bias: Float, val biasPrecision: Float, val step: Long, val updaterData: Array<FloatArray>) : BanditData {
    override fun migrate(from: IntArray, to: IntArray): BanditData {
        TODO("not implemented")
    }
}

