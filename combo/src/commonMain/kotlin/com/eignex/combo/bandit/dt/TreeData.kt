package com.eignex.combo.bandit.dt

import com.eignex.combo.shared.LearnerData
import com.eignex.combo.shared.SlotRemap
import com.eignex.kumulant.stat.regression.tree.ForestRegressionResult
import com.eignex.kumulant.stat.regression.tree.TreeRegressionResult

/**
 * Kumulant tree snapshot with the typed predicates defining its feature layout.
 *
 * @property state Immutable statistical snapshot.
 * @property splits Ordered typed predicates defining the vector coordinates.
 */
class DecisionTreeData internal constructor(val state: TreeRegressionResult, val splits: List<Split>) : LearnerData {
    override fun remap(slots: SlotRemap): LearnerData = throw UnsupportedOperationException(
        "Tree feature provenance cannot be remapped; rebuild the layout and retrain.",
    )
}

/**
 * Kumulant forest snapshot with the typed predicates defining its feature layout.
 *
 * @property state Immutable statistical snapshot.
 * @property splits Ordered typed predicates defining the vector coordinates.
 */
class ForestData internal constructor(val state: ForestRegressionResult, val splits: List<Split>) : LearnerData {
    override fun remap(slots: SlotRemap): LearnerData = throw UnsupportedOperationException(
        "Tree feature provenance cannot be remapped; rebuild the layout and retrain.",
    )
}
