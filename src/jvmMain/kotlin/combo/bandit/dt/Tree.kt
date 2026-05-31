package combo.bandit.dt

import com.eignex.kumulant.bandit.univariate.BanditPolicy
import com.eignex.kumulant.core.Result
import com.eignex.kumulant.stat.regression.tree.RegressionLeafNode
import com.eignex.kumulant.stat.regression.tree.RegressionNode
import com.eignex.kumulant.stat.regression.tree.RegressionSplitNode
import com.eignex.kumulant.stat.regression.tree.RegressionTree
import com.eignex.kumulant.stat.regression.tree.RegressionTreeConfig
import com.eignex.kumulant.stat.regression.tree.aggregate
import com.eignex.kumulant.stat.summary.MomentsResult
import com.eignex.kumulant.stat.summary.VarianceStat
import com.eignex.kumulant.stat.summary.WeightedMeanResult
import com.eignex.kumulant.stat.summary.WeightedVarianceResult
import kotlin.random.Random

/**
 * combo's online decision tree: a thin wrapper over kumulant's generic
 * [RegressionTree] specialised to a klause-coupled [TreeRow] via combo's [Split]
 * (which implements kumulant's `Split<TreeRow>` SPI). Growth — audit leaves, the
 * Hoeffding-bound split decision, mtry random subspaces, the variance-reduction
 * metric — lives entirely in kumulant now; combo adds only the bandit-facing walk and
 * descent helpers that score nodes via a [policy] and translate routes into klause
 * assumptions.
 *
 * Leaf arms are minted by the [policy] (`leafArmFactory = { policy.createArm() }`), so
 * the policy's priors seed every leaf exactly as before the kumulant migration.
 * Internal split nodes hold no live arm; they are scored on demand through kumulant's
 * exact per-node [aggregate] (Chan-merge of the subtree's leaves), which is
 * mathematically identical to the live internal arm combo used to keep.
 */
class Tree(
    val policy: BanditPolicy<WeightedVarianceResult>,
    candidates: List<Split>,
    config: RegressionTreeConfig = RegressionTreeConfig(),
    randomSeed: Int = 0,
) {
    private val inner: RegressionTree<TreeRow> = RegressionTree(
        splitCandidates = candidates,
        config = config,
        leafArmFactory = { policy.createArm() },
        randomSeed = randomSeed,
    )

    /** Fold an observation into the tree, possibly growing it. */
    fun update(row: TreeRow, reward: Double, weight: Double = 1.0) = inner.update(row, reward, weight)

    /** Walk to the leaf [row] resolves to. */
    fun findLeaf(row: TreeRow): RegressionLeafNode<TreeRow> = inner.findLeaf(row)

    /** Mean of the leaf [row] resolves to. */
    fun predict(row: TreeRow): Double = inner.findLeaf(row).arm.read(0L).mean

    /** Render the tree as nested `if (split) { ... } else { ... }` text. */
    fun prettyPrint(indent: String = ""): String = inner.prettyPrint(indent)

    /**
     * Descend through every [BoolSplit] whose direction is determined by [pinned] (a
     * map keyed by klause bool-variable id). Stops at the deepest reachable node: a
     * leaf (every split on the way down was pinned) or an unfixed split node — the
     * frontier the cross-tree forest algorithm scores. Non-bool splits stop the
     * descent (the caller sees them as frontier nodes).
     */
    fun descendTo(pinned: Map<Int, Boolean>, boolIdByName: Map<String, Int>): RegressionNode<TreeRow> {
        var node: RegressionNode<TreeRow> = inner.rootNode()
        while (node is RegressionSplitNode) {
            val direction = when (val split = node.split) {
                is BoolSplit -> boolIdByName[split.handle.name]?.let(pinned::get)
                else -> null
            }
            node = when (direction) {
                true -> node.pos
                false -> node.neg
                null -> return node
            }
        }
        return node
    }

    /**
     * Canonical walk-down: at each split, score both children's subtree aggregates via
     * [policy] (one fresh draw each — Thompson sampling per branching decision when
     * [policy] is `ThompsonSampling`), descend into the higher, end at a leaf. Returns
     * the chosen leaf plus the path of splits taken to reach it.
     */
    fun chooseLeaf(rng: Random, step: Long, maximize: Boolean): LeafChoice {
        val steps = mutableListOf<PathStep>()
        var node: RegressionNode<TreeRow> = inner.rootNode()
        while (node is RegressionSplitNode) {
            val sPos = signed(policy.evaluate(node.pos.aggregate(), step, rng), maximize)
            val sNeg = signed(policy.evaluate(node.neg.aggregate(), step, rng), maximize)
            val takePos = sPos >= sNeg
            steps += PathStep(node.split as Split, takePos)
            node = if (takePos) node.pos else node.neg
        }
        return LeafChoice(node as RegressionLeafNode<TreeRow>, steps)
    }

    /** Variant of [chooseLeaf] that follows the deterministic mean at each split —
     *  the "argmax exploit-only" path used by `optimalOrThrow`. */
    fun optimalLeaf(maximize: Boolean): LeafChoice {
        val steps = mutableListOf<PathStep>()
        var node: RegressionNode<TreeRow> = inner.rootNode()
        while (node is RegressionSplitNode) {
            val mPos = signed(node.pos.aggregate().mean, maximize)
            val mNeg = signed(node.neg.aggregate().mean, maximize)
            val takePos = mPos >= mNeg
            steps += PathStep(node.split as Split, takePos)
            node = if (takePos) node.pos else node.neg
        }
        return LeafChoice(node as RegressionLeafNode<TreeRow>, steps)
    }

    private fun signed(m: Double, maximize: Boolean): Double = if (maximize) m else -m

    companion object {
        /**
         * Merge subtree aggregates across trees into one weighted-variance result.
         * Used by [RandomForestBandit] to combine the per-tree frontier statistics for
         * a candidate direction before scoring it.
         */
        fun mergeAggregates(snaps: Iterable<WeightedVarianceResult>): WeightedVarianceResult {
            val combined = VarianceStat()
            for (s in snaps) combined.merge(s)
            return combined.read(0L)
        }
    }
}

/** One step of a root-to-leaf path: which [Split] routed here, and the side taken. */
data class PathStep(val split: Split, val tookPos: Boolean)

/** Walk result: the chosen kumulant leaf node plus the sequence of splits taken to it. */
data class LeafChoice(val leaf: RegressionLeafNode<TreeRow>, val path: List<PathStep>)

/** Read a scalar mean from any of the kumulant Result types used as bandit-arm
 *  sufficient statistics. Used by both DT and RF for prediction and scoring. */
internal fun scalarMean(snapshot: Result): Double = when (snapshot) {
    is WeightedVarianceResult -> snapshot.mean
    is WeightedMeanResult -> snapshot.mean
    is MomentsResult -> snapshot.mean
    is com.eignex.kumulant.stat.summary.BernoulliSumResult ->
        if (snapshot.trials <= 0.0) 0.0 else snapshot.successes / snapshot.trials
    else -> error("Tree can't read mean from snapshot type ${snapshot::class}")
}
