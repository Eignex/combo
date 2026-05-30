package combo.bandit.glm

import com.eignex.kumulant.stat.regression.glm.LinearRegressionResult
import combo.bandit.LearnerData
import combo.bandit.SlotRemap
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Combo-side wrapper around a kumulant [LinearRegressionResult] that satisfies the
 * [LearnerData] contract. The snapshot's `weights` are indexed by the linear
 * bandit's feature layout — schema changes invalidate that mapping, so [remap] is
 * a hard error: rebuild the feature index, call [com.eignex.kumulant.core.Stat.reset]
 * on the underlying regression stat, and re-train instead.
 */
@Serializable
@SerialName("LinearLearnerData")
data class LinearLearnerData(val state: LinearRegressionResult) : LearnerData {
    override fun remap(slots: SlotRemap): LinearLearnerData =
        throw NotImplementedError(
            "LinearLearnerData.remap requires bandit-specific feature provenance; " +
                "reset the regression stat and retrain instead."
        )
}
