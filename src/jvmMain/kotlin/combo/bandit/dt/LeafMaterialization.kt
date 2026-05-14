package combo.bandit.dt

import com.eignex.klause.ast.And
import com.eignex.klause.ast.BoolExpr
import com.eignex.klause.ast.BoolRef
import com.eignex.klause.ast.NominalEq
import com.eignex.klause.ast.Not
import com.eignex.klause.solver.Assumptions
import com.eignex.klause.solver.Sample
import combo.decisions.BanditSample
import combo.decisions.CompiledDecisionSpace
import kotlin.random.Random

/**
 * Translate a tree's root-to-leaf [path] into the strongest klause [Assumptions] we
 * can express plus the residual checks that have to be enforced post-sample.
 *
 *  - [BoolSplit] / one-vs-rest [NominalSplit] (single-label `leftLabels`) on the
 *    "pos" side pin a single bool indicator and can ride into the assumptions map.
 *  - [IntThresholdSplit] and many-label nominal splits encode a *range* of values
 *    that klause's value-pin assumptions can't express. Those stay as residual
 *    predicates evaluated against the encoded [TreeRow].
 *
 * The bandit calls this once per choose round, then rejection-samples against
 * [Materialization.residual] until a sample satisfies the whole path (or the
 * retry budget runs out).
 */
internal data class Materialization(
    val assumptions: Assumptions,
    val residual: List<(TreeRow) -> Boolean>,
)

internal fun materialize(
    space: CompiledDecisionSpace,
    path: List<PathStep>,
): Materialization {
    val bools = mutableMapOf<Int, Boolean>()
    val residual = mutableListOf<(TreeRow) -> Boolean>()
    for (step in path) {
        val split = step.split
        val took = step.tookPos
        when (split) {
            is BoolSplit -> {
                val id = space.compiled.boolVarIdByName[split.handle.name]
                if (id != null) bools[id] = took
                else residual += { row -> split.direction(row) == took }
            }
            is NominalSplit -> {
                // One-vs-rest by design — pins the indicator bit directly. `took=true`
                // means value == label → indicator true. `took=false` → indicator false.
                val indicators = space.compiled.nominalIndicators[split.handle.name]
                val bitId = indicators?.get(split.label)
                if (bitId != null) {
                    bools[bitId] = took
                } else {
                    residual += { row -> split.direction(row) == took }
                }
            }
            is IntThresholdSplit -> {
                // Range pin — klause's assumption format doesn't carry ranges, so the
                // tree filters via rejection.
                residual += { row -> split.direction(row) == took }
            }
            is FloatThresholdSplit -> {
                // Same story as int thresholds: klause assumptions are value pins, not
                // range pins. The split's [direction] decodes to real-value Double and
                // compares against the threshold, so the filter is correct regardless
                // of klause's bucket count.
                residual += { row -> split.direction(row) == took }
            }
            is ExprSplit -> {
                // Walk the expression for top-level pinnable atoms: a top-level BoolRef
                // or NominalEq (or their negations / conjunctions of such) can pin
                // single-value assumptions; everything else stays residual.
                pinExprAtoms(split.expr, took, space, bools, residual)
            }
        }
    }
    return Materialization(Assumptions(bools, emptyMap()), residual)
}

/**
 * Draw samples from [proposeSample] until one satisfies [residual] and the encoder
 * agrees its row routes to [target] under the same tree's split structure. Returns
 * `null` once [budget] attempts fail.
 *
 * The leaf-identity check (`tree.findLeaf(row) === target`) is the canonical
 * acceptance test: it covers every path constraint at once, including those we
 * already pinned into [assumptions]. We still check [residual] first because it's
 * cheap and short-circuits the more expensive leaf walk on early misses.
 */
internal inline fun materializeLeaf(
    rng: Random,
    budget: Int,
    space: CompiledDecisionSpace,
    proposeSample: (Random, Assumptions) -> Sample?,
    assumptions: Assumptions,
    residual: List<(TreeRow) -> Boolean>,
    encode: (BanditSample) -> TreeRow,
    routesTo: (TreeRow) -> Boolean,
): BanditSample? {
    var i = 0
    while (i < budget) {
        val s = proposeSample(rng, assumptions)
        if (s != null) {
            // Dither inside the loop: routing checks must use the same continuous
            // float values the bandit will later train on. Otherwise a candidate's
            // bucket midpoint may agree with the chosen leaf while its dithered
            // value disagrees, biasing training.
            val wrapped = BanditSample.dithered(s, space, rng)
            val row = encode(wrapped)
            if (residual.all { it(row) } && routesTo(row)) return wrapped
        }
        i++
    }
    return null
}

/**
 * Walk [expr] under the assumption that it must evaluate to [took]. Whenever a
 * conjunctive atom can be expressed as a single-value pin in klause [Assumptions]
 * — i.e. a [BoolRef] or [NominalEq], possibly under a [Not], possibly under a
 * conjunction of those — pin it into [bools]. Anything that doesn't decompose
 * cleanly falls back to a residual predicate evaluating the whole expression
 * against the candidate row.
 *
 * The walk only descends into [And] when [took] is true (a top-level conjunction
 * forces every conjunct to be true) and into [Not] flipping the sign. Disjunctions
 * and other compound forms can't pin without case-splitting; they stay residual.
 */
private fun pinExprAtoms(
    expr: BoolExpr,
    took: Boolean,
    space: CompiledDecisionSpace,
    bools: MutableMap<Int, Boolean>,
    residual: MutableList<(TreeRow) -> Boolean>,
) {
    if (tryPinAtom(expr, took, space, bools)) return
    if (expr is And && took) {
        for (child in expr.children) pinExprAtoms(child, true, space, bools, residual)
        return
    }
    if (expr is Not) {
        pinExprAtoms(expr.child, !took, space, bools, residual)
        return
    }
    residual += { row -> space.evaluate(expr, row.sample.sample) == took }
}

/** Try to pin a single bool variable from [expr] given the target [took]. Returns
 *  true on success. Handles `BoolRef` (with negation), `Not(BoolRef)`, and `NominalEq`
 *  (pins the indicator bit). */
private fun tryPinAtom(
    expr: BoolExpr,
    took: Boolean,
    space: CompiledDecisionSpace,
    bools: MutableMap<Int, Boolean>,
): Boolean = when (expr) {
    is BoolRef -> {
        val id = space.compiled.boolVarIdByName[expr.name] ?: return false
        bools[id] = took xor expr.negated
        true
    }
    is Not -> {
        // Recurse with flipped sign.
        tryPinAtom(expr.child, !took, space, bools)
    }
    is NominalEq -> {
        val bitId = space.compiled.nominalIndicators[expr.name]?.get(expr.label) ?: return false
        bools[bitId] = took
        true
    }
    else -> false
}
