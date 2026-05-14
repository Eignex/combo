package combo.bandit.dt

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
                // pos=true means value ∈ leftLabels. One-vs-rest (single label) pins
                // its indicator bit directly; the inverted "value ∉ leftLabels" with
                // a single label pins the indicator to false. Multi-label partitions
                // collapse to a residual predicate.
                if (split.leftLabels.size == 1) {
                    val label = split.leftLabels.single()
                    val indicators = space.compiled.nominalIndicators[split.handle.name]
                    val bitId = indicators?.get(label)
                    if (bitId != null) {
                        bools[bitId] = took
                    } else {
                        residual += { row -> split.direction(row) == took }
                    }
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
