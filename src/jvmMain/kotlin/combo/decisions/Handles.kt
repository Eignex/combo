package combo.decisions

import com.eignex.klause.ast.And
import com.eignex.klause.ast.AtLeast
import com.eignex.klause.ast.AtMost
import com.eignex.klause.ast.BoolExpr
import com.eignex.klause.ast.BoolTerm
import com.eignex.klause.ast.CardinalityExpr
import com.eignex.klause.ast.IntExpr
import com.eignex.klause.ast.IntIfThenElse
import com.eignex.klause.ast.IntLit
import com.eignex.klause.ast.IntSum
import com.eignex.klause.ast.IntTerm
import com.eignex.klause.ast.Or
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

/**
 * Multi-select decision variable: a set of [labels], each backed by its own bool
 * indicator. From the schema's perspective `multiple("a", "b", "c")` is exactly N
 * independent bools registered under `<name>.<label>`; the handle adds typed
 * `contains` / `containsAny` / `containsAll` / `sizeGe` / etc. operators so user
 * constraints read like the [com.eignex.klause.schema.NominalHandle] vocabulary.
 *
 * The bandit pipeline treats the underlying bools as regular features (split
 * candidates, linear coefficients, …) — nothing in the search/learning code needs
 * to know about the multi-select grouping.
 */
class MultipleHandle internal constructor(
    val name: String,
    val labels: List<String>,
    /** Per-label bool indicator. Key set is [labels]; the values' [BoolHandle.name]s
     *  are `"$name.$label"`. */
    val bools: Map<String, BoolHandle>,
) {
    init {
        require(labels.isNotEmpty()) { "Multiple '$name' must have at least one label" }
    }

    /** Indicator BoolExpr: true iff [label] is selected. */
    fun contains(label: String): BoolExpr {
        require(label in labels) { "Label '$label' not in multiple '$name' (have $labels)" }
        return bools.getValue(label).toExpr()
    }

    /** OR over the given labels' indicators. */
    fun containsAny(vararg ls: String): BoolExpr {
        require(ls.isNotEmpty()) { "containsAny needs at least one label" }
        ls.forEach { require(it in labels) { "Label '$it' not in multiple '$name'" } }
        val terms = ls.map { bools.getValue(it).toExpr() }
        return if (terms.size == 1) terms[0] else Or(terms)
    }

    /** AND over the given labels' indicators. */
    fun containsAll(vararg ls: String): BoolExpr {
        require(ls.isNotEmpty()) { "containsAll needs at least one label" }
        ls.forEach { require(it in labels) { "Label '$it' not in multiple '$name'" } }
        val terms = ls.map { bools.getValue(it).toExpr() }
        return if (terms.size == 1) terms[0] else And(terms)
    }

    /** Cardinality: at least [k] labels selected. */
    fun sizeGe(k: Int): BoolExpr = AtLeast(allIndicators(), k)

    /** Cardinality: at most [k] labels selected. */
    fun sizeLe(k: Int): BoolExpr = AtMost(allIndicators(), k)

    /** Cardinality: between [min] and [max] (inclusive) labels selected. */
    fun sizeBetween(min: Int, max: Int): BoolExpr = CardinalityExpr(allIndicators(), min, max)

    /** Cardinality: exactly [k] labels selected. */
    fun sizeEq(k: Int): BoolExpr = sizeBetween(k, k)

    /** Number-of-selected as an [IntExpr] — useful when summing across multiples. */
    fun count(): IntExpr =
        IntSum(allIndicators().map { IntIfThenElse(it, IntLit(1), IntLit(0)) })

    private fun allIndicators(): List<BoolExpr> = labels.map { bools.getValue(it).toExpr() }

    override fun equals(other: Any?): Boolean = other is MultipleHandle && other.name == name
    override fun hashCode(): Int = name.hashCode()
    override fun toString(): String = "MultipleHandle($name, labels=$labels)"
}
