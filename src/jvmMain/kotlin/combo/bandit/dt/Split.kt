package combo.bandit.dt

import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.IntHandle
import com.eignex.klause.schema.NominalHandle

/**
 * A predicate that routes a [TreeRow] to the "left" (true) or "right" (false) branch of
 * a decision tree. Sealed so split-metric code and pruning logic can dispatch over a
 * known closed set; concrete subtypes are keyed on klause handle types directly.
 *
 * Slice 1 covers the basic splits. Future slices add:
 *  - `IsPresentSplit` for routing on an optional variable's active state.
 *  - Quantitative `IntThresholdSplit` variants (`<`, `<=`, etc.).
 *  - Nominal splits over non-trivial label partitions (this slice supports any subset).
 */
sealed interface Split {
    /** Direction of [row]: true → left branch, false → right branch. */
    fun direction(row: TreeRow): Boolean
}

/** Route by a bool handle's value. */
data class BoolSplit(val handle: BoolHandle) : Split {
    override fun direction(row: TreeRow): Boolean = row.bool(handle)
}

/** Route by `int <= threshold`. Threshold is inclusive on the "left" side. */
data class IntThresholdSplit(val handle: IntHandle, val threshold: Int) : Split {
    init {
        require(threshold in handle.min..handle.max) {
            "threshold $threshold outside the int handle '${handle.name}' domain [${handle.min}, ${handle.max}]"
        }
    }
    override fun direction(row: TreeRow): Boolean = row.int(handle) <= threshold
}

/**
 * Route by membership of the nominal's value in [leftLabels]. The complementary set
 * of [handle].labels takes the right branch. [leftLabels] must be a non-trivial subset
 * (neither empty nor the full label set) — otherwise no row is ever split.
 */
data class NominalSplit(val handle: NominalHandle, val leftLabels: Set<String>) : Split {
    init {
        require(leftLabels.isNotEmpty() && leftLabels.size < handle.labels.size) {
            "leftLabels must partition '${handle.name}' into two non-empty groups; got $leftLabels"
        }
        require(handle.labels.containsAll(leftLabels)) {
            "leftLabels $leftLabels are not all present in nominal '${handle.name}' labels ${handle.labels}"
        }
    }
    override fun direction(row: TreeRow): Boolean = row.nominal(handle) in leftLabels
}
