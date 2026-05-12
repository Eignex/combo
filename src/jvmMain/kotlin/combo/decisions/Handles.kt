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
 */
class BoolContextHandle internal constructor(val klauseHandle: BoolHandle) : BoolTerm {
    val name: String get() = klauseHandle.name
    override fun toExpr(): BoolExpr = klauseHandle.toExpr()

    override fun equals(other: Any?): Boolean = other is BoolContextHandle && other.name == name
    override fun hashCode(): Int = name.hashCode()
    override fun toString(): String = "BoolContextHandle($name)"
}

/**
 * Contextual integer feature — supplied by the caller at choose/update time. Backed by
 * a real solver variable so it participates in `constraint { … }` like any decision
 * variable, but its value is fixed per call via an assumption rather than searched.
 */
class IntContextHandle internal constructor(val klauseHandle: IntHandle) : IntTerm {
    val name: String get() = klauseHandle.name
    val min: Int get() = klauseHandle.min
    val max: Int get() = klauseHandle.max
    override fun toIntExpr(): IntExpr = klauseHandle.toIntExpr()

    override fun equals(other: Any?): Boolean = other is IntContextHandle && other.name == name
    override fun hashCode(): Int = name.hashCode()
    override fun toString(): String = "IntContextHandle($name)"
}
