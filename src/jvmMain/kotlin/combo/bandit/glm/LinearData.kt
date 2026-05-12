package combo.bandit.glm

import combo.bandit.BanditData
import combo.bandit.SlotRemap
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Serializable snapshot of a [LinearModel]: feature-indexed weights, intercept,
 * the SGD/covariance updater's auxiliary state, and the bookkeeping step counter.
 *
 * The feature index is bandit-defined — it depends on how the wrapping
 * [combo.bandit.PredictionBandit] projects a klause `Sample` into a feature vector.
 * Because of that, [remap] is intentionally a no-op contract today: when the
 * schema changes, the wrapping bandit must rebuild the feature index and call
 * `blank()` on the model, then re-train. A future revision can carry feature
 * provenance here and do honest remapping.
 */
@Serializable
@SerialName("LinearData")
data class LinearData(
    val weights: FloatArray,
    val bias: Float,
    val biasPrecision: Float,
    val step: Long,
    val updaterData: List<FloatArray>,
) : BanditData {
    override fun remap(slots: SlotRemap): LinearData =
        throw NotImplementedError(
            "LinearData.remap requires bandit-specific feature provenance; rebuild via LinearModel.blank() and retrain instead."
        )

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is LinearData) return false
        return bias == other.bias &&
            biasPrecision == other.biasPrecision &&
            step == other.step &&
            weights.contentEquals(other.weights) &&
            updaterData.size == other.updaterData.size &&
            updaterData.indices.all { updaterData[it].contentEquals(other.updaterData[it]) }
    }

    override fun hashCode(): Int {
        var result = weights.contentHashCode()
        result = 31 * result + bias.hashCode()
        result = 31 * result + biasPrecision.hashCode()
        result = 31 * result + step.hashCode()
        for (row in updaterData) result = 31 * result + row.contentHashCode()
        return result
    }
}
