package combo.bandit.dt

import com.eignex.kumulant.core.Result
import com.eignex.kumulant.core.SeriesStat

/**
 * Internal tree node. Every node — both internal splits and leaves — carries an
 * arm tracking observations that flowed through it. This is the canonical
 * "tree-partitions-arms" model: a [SplitNode]'s arm is the aggregate over its
 * subtree, used by the bandit's choose-time walk-down to score branching
 * decisions; a [LeafNode]'s arm is the per-arm sufficient statistic.
 */
sealed class Node<R : Result> {
    /** Per-subtree arm. For leaves this is *the* arm; for split nodes it's the
     *  aggregate stat over every observation routed through the node, used for
     *  policy-driven walk-down at choose time. */
    abstract val arm: SeriesStat<R>

    /** Walk to the leaf this row resolves to. */
    abstract fun findLeaf(row: TreeRow): LeafNode<R>
}

/**
 * Routes by [split] to either [pos] (split-true) or [neg] (split-false). The
 * aggregate [arm] is updated by the tree on every observation flowing through.
 */
class SplitNode<R : Result>(
    val split: Split,
    var pos: Node<R>,
    var neg: Node<R>,
    override val arm: SeriesStat<R>,
) : Node<R>() {
    override fun findLeaf(row: TreeRow): LeafNode<R> =
        if (split.direction(row)) pos.findLeaf(row) else neg.findLeaf(row)
}

/**
 * Holds the per-arm accumulator at a path. Two flavours:
 *
 *  - [AuditLeaf] — actively considers candidate splits each round and may convert to a
 *    [SplitNode]. Tracks pos/neg arm stats per candidate.
 *  - [TerminalLeaf] — frozen at max depth or when no candidates remain.
 */
sealed class LeafNode<R : Result> : Node<R>() {
    final override fun findLeaf(row: TreeRow): LeafNode<R> = this
}

/** Frozen leaf — no further splits will be considered. */
class TerminalLeaf<R : Result>(override val arm: SeriesStat<R>) : LeafNode<R>()

/**
 * Leaf that tracks per-candidate pos/neg stats. When a candidate clears the
 * Hoeffding-bound test in the tree's update loop, this leaf is replaced by a
 * [SplitNode]. The candidate subset is per-leaf — picked at leaf birth — so
 * mtry-style random subspace selection lives at the leaf level.
 */
class AuditLeaf<R : Result>(
    override val arm: SeriesStat<R>,
    val candidates: List<Split>,
    val pos: List<SeriesStat<R>>,
    val neg: List<SeriesStat<R>>,
) : LeafNode<R>() {
    init {
        require(candidates.size == pos.size && pos.size == neg.size) {
            "candidates/pos/neg must align: ${candidates.size}/${pos.size}/${neg.size}"
        }
    }
    var observationsSinceLastCheck: Int = 0
}
