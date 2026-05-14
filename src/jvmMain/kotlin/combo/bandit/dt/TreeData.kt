package combo.bandit.dt

import combo.bandit.LearnerData
import combo.bandit.SlotRemap

/**
 * Serialisable snapshot of a [DecisionTreeBandit]. Slice 1 is a no-op placeholder —
 * the tree state isn't yet round-trippable; a follow-up will design the wire format
 * once the [Split] hierarchy stabilises (in particular: how to serialise klause
 * handles vs. their qualified names).
 *
 * Concrete bandits still implement `importData` / `exportData` against this type so
 * future state-transfer support drops in without an interface change.
 */
class DecisionTreeData internal constructor() : LearnerData {
    override fun remap(slots: SlotRemap): LearnerData = this
}

/** Placeholder for the eventual forest data. Same caveats as [DecisionTreeData]. */
class ForestData internal constructor(val trees: List<DecisionTreeData>) : LearnerData {
    override fun remap(slots: SlotRemap): LearnerData = this
}
