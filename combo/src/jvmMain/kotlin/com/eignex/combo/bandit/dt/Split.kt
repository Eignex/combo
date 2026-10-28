package com.eignex.combo.bandit.dt

import com.eignex.klause.util.bigIntOf
import com.eignex.klause.util.compareTo
import com.eignex.klause.model.And
import com.eignex.klause.model.AtLeast
import com.eignex.klause.model.AtMost
import com.eignex.klause.model.BoolExpr
import com.eignex.klause.model.BoolRef
import com.eignex.klause.model.CardinalityExpr
import com.eignex.klause.model.Iff
import com.eignex.klause.model.Implies
import com.eignex.klause.model.IntAbs
import com.eignex.klause.model.IntCmpOp
import com.eignex.klause.model.IntCompare
import com.eignex.klause.model.IntExpr
import com.eignex.klause.model.IntLit
import com.eignex.klause.model.IntMax
import com.eignex.klause.model.IntMin
import com.eignex.klause.model.IntRef
import com.eignex.klause.model.IntScale
import com.eignex.klause.model.IntSum
import com.eignex.klause.model.NominalEq
import com.eignex.klause.model.Not
import com.eignex.klause.model.Or
import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.FloatHandle
import com.eignex.klause.schema.IntHandle
import com.eignex.klause.schema.NominalHandle
import com.eignex.klause.schema.le
import com.eignex.kumulant.stat.regression.tree.Split as RowSplit

/** Resolve predicates already represented by a Boolean coordinate in Klause. */
internal fun Split.booleanId(boolIdByName: Map<String, Int>, nominalIndicators: Map<String, Map<String, Int>>): Int? =
    when (this) {
        is BoolSplit -> boolIdByName[handle.name]
        is NominalSplit -> nominalIndicators[handle.name]?.get(label)
        else -> null
    }

/**
 * Binary predicate routing a [TreeRow] to "left" (true) or "right" (false). Every
 * split carries a klause [BoolExpr] — the same vocabulary users already write inside
 * `constraint { ... }` blocks — so splits and user constraints share one language,
 * one evaluator, one pretty printer.
 *
 * The typed cases ([BoolSplit] / [IntThresholdSplit] / [NominalSplit] /
 * [FloatThresholdSplit]) cover the common shapes and let [materializeLeaf]
 * dispatch on them to translate splits into klause [com.eignex.klause.propagation.Assumptions].
 * [ExprSplit] is the escape hatch for arbitrary expressions — `tier eq "pro" and budget gt 100`
 * is a valid split.
 */
sealed interface Split : RowSplit<TreeRow> {
    /** The klause expression that defines this split's routing predicate. */
    val expr: BoolExpr

    /** Evaluate the predicate against [row]. Default uses the shared
     *  [com.eignex.combo.decisions.CompiledDecisionSpace.evaluate]; concrete cases may override
     *  with a tighter implementation when it's cheaper. Implements kumulant's
     *  [RowSplit] growth-time routing SPI, so combo's klause-coupled splits plug
     *  directly into kumulant's generic tree engine over a [TreeRow]. */
    override fun direction(row: TreeRow): Boolean = row.space.evaluate(expr, row.sample.sample)

    override fun toString(): String
}

/** Route by a bool handle's value. */
data class BoolSplit(val handle: BoolHandle) : Split {
    override val expr: BoolExpr = BoolRef(handle.name)
    override fun direction(row: TreeRow): Boolean = row.bool(handle)
    override fun toString(): String = handle.name
}

/** Route by `int <= threshold`. Threshold is inclusive on the "left" side. */
data class IntThresholdSplit(val handle: IntHandle, val threshold: Long) : Split {
    init {
        require(threshold in handle.min..handle.max) {
            "threshold $threshold outside int handle '${handle.name}' domain [${handle.min}, ${handle.max}]"
        }
    }
    override val expr: BoolExpr = if (threshold in Int.MIN_VALUE.toLong()..Int.MAX_VALUE.toLong()) {
        handle le threshold.toInt()
    } else com.eignex.klause.schema.factor(
        com.eignex.klause.factor.arithmetic.Linear(
            longArrayOf(1), intArrayOf(0), com.eignex.klause.ir.LinearOp.LE, threshold,
        ), ints = listOf(handle),
    )
    override fun direction(row: TreeRow): Boolean = if (row.space.coordinates.schema == null) {
        row.sample.sample.exactIntValue(row.space.coordinates.intVarIdByName.getValue(handle.name)) <= bigIntOf(threshold)
    } else row.int(handle) <= threshold
    override fun toString(): String = "${handle.name} <= $threshold"
}

/**
 * Route by `nominal == label` (one-vs-rest). Multi-label partitions can be expressed
 * by wrapping an `Or` expression in [ExprSplit].
 */
data class NominalSplit(val handle: NominalHandle, val label: String) : Split {
    init {
        require(label in handle.labels) {
            "label '$label' not in nominal '${handle.name}' labels ${handle.labels}"
        }
    }
    override val expr: BoolExpr = handle eq label
    override fun direction(row: TreeRow): Boolean = row.nominal(handle) == label
    override fun toString(): String = "${handle.name} == \"$label\""
}

/**
 * Route by `floatValue <= threshold` in **real-value Double space**. See
 * `FloatThresholdSplit` documentation for the bucket-independence story.
 */
data class FloatThresholdSplit(val handle: FloatHandle, val threshold: Double) : Split {
    init {
        require(threshold in handle.min..handle.max) {
            "threshold $threshold outside float handle '${handle.name}' domain [${handle.min}, ${handle.max}]"
        }
    }
    override val expr: BoolExpr = handle le threshold
    override fun direction(row: TreeRow): Boolean = row.float(handle) <= threshold
    override fun toString(): String = "${handle.name} <= $threshold"
}

/**
 * Arbitrary klause expression as a split predicate. Use when no typed case fits — for
 * example a multi-variable conjunction, an inequality between two ints, or a cardinality
 * constraint. [materializeLeaf] pins what it can (top-level [BoolRef] / [NominalEq]
 * conjuncts) and treats the rest as residual rejection-sampled predicates.
 */
data class ExprSplit(override val expr: BoolExpr) : Split {
    override fun toString(): String = expr.pretty()
}

/**
 * Render a klause [BoolExpr] in user-readable infix form. Mirrors the syntax users
 * write in `constraint { ... }` so split prints look like the constraints they came
 * from. Falls through to the data-class `toString()` for forms without a tidy
 * surface syntax.
 */
fun BoolExpr.pretty(): String = when (this) {
    is BoolRef -> if (negated) "!$name" else name
    is NominalEq -> "$name == \"$label\""
    is Not -> "!${child.pretty().parenthesise()}"
    is And -> children.joinToString(" and ") { it.pretty().parenthesise() }
    is Or -> children.joinToString(" or ") { it.pretty().parenthesise() }
    is Implies -> "${left.pretty().parenthesise()} -> ${right.pretty().parenthesise()}"
    is Iff -> "${left.pretty().parenthesise()} <-> ${right.pretty().parenthesise()}"
    is AtMost -> "atMost($k, ${children.joinToString(", ") { it.pretty() }})"
    is AtLeast -> "atLeast($k, ${children.joinToString(", ") { it.pretty() }})"
    is CardinalityExpr -> "cardinality($min..$max, ${children.joinToString(", ") { it.pretty() }})"
    is IntCompare -> "${left.pretty()} ${op.symbol} ${right.pretty()}"
    else -> toString()
}

private fun IntExpr.pretty(): String = when (this) {
    is IntRef -> name
    is IntLit -> value.toString()
    is IntScale -> "$coeff * ${child.pretty()}"
    is IntSum -> children.joinToString(" + ") { it.pretty() }
    is IntMin -> "min(${children.joinToString(", ") { it.pretty() }})"
    is IntMax -> "max(${children.joinToString(", ") { it.pretty() }})"
    is IntAbs -> "|${child.pretty()}|"
    else -> toString()
}

private val IntCmpOp.symbol: String
    get() = when (this) {
        IntCmpOp.EQ -> "=="
        IntCmpOp.NE -> "!="
        IntCmpOp.LE -> "<="
        IntCmpOp.LT -> "<"
        IntCmpOp.GE -> ">="
        IntCmpOp.GT -> ">"
    }

/** Wrap a sub-expression in parens iff it parses as a compound (heuristic: contains a space). */
private fun String.parenthesise(): String = if (contains(' ') && !startsWith('(')) "($this)" else this
