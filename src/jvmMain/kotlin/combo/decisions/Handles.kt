package combo.decisions

import com.eignex.klause.ast.BoolExpr
import com.eignex.klause.ast.BoolTerm
import com.eignex.klause.ast.IntExpr
import com.eignex.klause.ast.IntTerm
import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.IntHandle

/**
 * Contextual boolean feature — supplied by the caller at choose/update time. Backed by
 * a real solver variable so it participates in `constraint { … }` like any decision
 * variable, but its value is fixed per call via an assumption rather than searched.
 *
 * If [isKnownGate] is non-null this context is *optional*: the caller may indicate
 * absence by not calling `set(handle, …)` in the [Context] builder. When absent, the
 * `isKnownGate` variable is pinned to false (which transitively pins the value to its
 * default and masks the feature slot).
 */
class BoolContextHandle internal constructor(
    val klauseHandle: BoolHandle,
    internal val isKnownGate: BoolHandle? = null,
) : BoolTerm {
    val name: String get() = klauseHandle.name
    val isOptional: Boolean get() = isKnownGate != null
    override fun toExpr(): BoolExpr = klauseHandle.toExpr()

    override fun equals(other: Any?): Boolean = other is BoolContextHandle && other.name == name
    override fun hashCode(): Int = name.hashCode()
    override fun toString(): String = "BoolContextHandle($name${if (isOptional) ", optional" else ""})"
}

/**
 * Contextual integer feature — supplied by the caller at choose/update time. Backed by
 * a real solver variable so it participates in `constraint { … }` like any decision
 * variable, but its value is fixed per call via an assumption rather than searched.
 *
 * See [BoolContextHandle] for the optional-context semantics carried by [isKnownGate].
 */
class IntContextHandle internal constructor(
    val klauseHandle: IntHandle,
    internal val isKnownGate: BoolHandle? = null,
) : IntTerm {
    val name: String get() = klauseHandle.name
    val min: Int get() = klauseHandle.min
    val max: Int get() = klauseHandle.max
    val isOptional: Boolean get() = isKnownGate != null
    override fun toIntExpr(): IntExpr = klauseHandle.toIntExpr()

    override fun equals(other: Any?): Boolean = other is IntContextHandle && other.name == name
    override fun hashCode(): Int = name.hashCode()
    override fun toString(): String = "IntContextHandle($name${if (isOptional) ", optional" else ""})"
}
