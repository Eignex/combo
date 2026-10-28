package com.eignex.combo.bandit.dt

import com.eignex.klause.propagation.Assumptions
import com.eignex.koblas.Vector
import com.eignex.kumulant.bandit.univariate.BanditPolicy
import com.eignex.kumulant.stat.regression.tree.RegressionLeafNode
import com.eignex.kumulant.stat.regression.tree.RegressionNode
import com.eignex.kumulant.stat.regression.tree.RegressionSplitNode
import com.eignex.kumulant.stat.regression.tree.RegressionTree
import com.eignex.kumulant.stat.regression.tree.RegressionTreeConfig
import com.eignex.kumulant.stat.regression.tree.aggregate
import com.eignex.kumulant.stat.regression.tree.mergeForestEvidence
import com.eignex.kumulant.stat.summary.WeightedVarianceResult
import kotlin.random.Random

/**
 * Typed Klause adapter over a Kumulant vector tree. [TreeVectorLayout] preserves numeric
 * units and maps learned vector predicates back to their original Klause expressions.
 * Kumulant owns growth, leaf arms, statistics, and snapshots.
 */
class Tree internal constructor(
    /** Scorer for leaf and subtree summaries. */
    val policy: BanditPolicy<WeightedVarianceResult>,
    internal val layout: TreeVectorLayout,
    private val tree: () -> RegressionTree<Vector>,
) {
    /** Build a standalone tree using typed split candidates. */
    constructor(
        policy: BanditPolicy<WeightedVarianceResult>,
        candidates: List<Split>,
        config: RegressionTreeConfig = RegressionTreeConfig(),
        randomSeed: Int = 0,
    ) : this(policy, TreeVectorLayout(candidates), config, randomSeed)

    private constructor(
        policy: BanditPolicy<WeightedVarianceResult>,
        layout: TreeVectorLayout,
        config: RegressionTreeConfig,
        randomSeed: Int,
    ) : this(
        policy,
        layout,
        RegressionTree<Vector>(
            layout.splits,
            config,
            leafArmFactory = { policy.createArm() },
            randomSeed = randomSeed,
        ).let { inner -> { inner } },
    )

    internal val inner: RegressionTree<Vector> get() = tree()

    /** Fold an observation into the tree, possibly growing it. */
    fun update(row: TreeRow, reward: Double, weight: Double = 1.0) = inner.update(layout.encode(row), reward, weight)

    /** Walk to the leaf [row] resolves to. */
    fun findLeaf(row: TreeRow): RegressionLeafNode<Vector> = inner.findLeaf(layout.encode(row))

    /** Mean of the leaf [row] resolves to. */
    fun predict(row: TreeRow): Double = inner.findLeaf(layout.encode(row)).arm.read(0L).mean

    /** Render the tree as nested `if (split) { ... } else { ... }` text. */
    fun prettyPrint(indent: String = ""): String = layout.splits.fold(inner.prettyPrint(indent)) { text, split ->
        text.replace(split.toString(), layout.predicate(split).toString())
    }

    /**
     * Descend through every [BoolSplit] whose direction is determined by [pinned] (a
     * map keyed by klause bool-variable id). Stops at the deepest reachable node: a
     * leaf (every split on the way down was pinned) or an unfixed split node — the
     * frontier the cross-tree forest algorithm scores. Nominal indicator pins and
     * exact integer context values also determine routes. Other numeric splits
     * remain frontier nodes and participate in the hybrid's candidate scoring.
     */
    fun descendTo(
        pinned: Map<Int, Boolean>,
        boolIdByName: Map<String, Int>,
        nominalIndicators: Map<String, Map<String, Int>> = emptyMap(),
        assumptions: Assumptions = Assumptions.None,
        intIdByName: Map<String, Int> = emptyMap(),
    ): RegressionNode<Vector> {
        var node: RegressionNode<Vector> = inner.rootNode()
        while (node is RegressionSplitNode) {
            val direction = when (val split = layout.predicate(node.split)) {
                is BoolSplit -> boolIdByName[split.handle.name]?.let(pinned::get)

                is NominalSplit -> nominalIndicators[split.handle.name]?.get(split.label)?.let(pinned::get)

                is IntThresholdSplit -> intIdByName[split.handle.name]?.let { id ->
                    assumptions.intValueOrNull(id)?.let { it <= split.threshold }
                }

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
        var node: RegressionNode<Vector> = inner.rootNode()
        while (node is RegressionSplitNode) {
            val sPos = signed(policy.evaluate(node.pos.aggregate(), step, rng), maximize)
            val sNeg = signed(policy.evaluate(node.neg.aggregate(), step, rng), maximize)
            val takePos = sPos >= sNeg
            steps += PathStep(layout.predicate(node.split), takePos)
            node = if (takePos) node.pos else node.neg
        }
        return LeafChoice(node as RegressionLeafNode<Vector>, steps)
    }

    /** Variant of [chooseLeaf] that follows the deterministic mean at each split —
     *  the "argmax exploit-only" path used by `optimalOrThrow`. */
    fun optimalLeaf(maximize: Boolean): LeafChoice {
        val steps = mutableListOf<PathStep>()
        var node: RegressionNode<Vector> = inner.rootNode()
        while (node is RegressionSplitNode) {
            val mPos = signed(node.pos.aggregate().mean, maximize)
            val mNeg = signed(node.neg.aggregate().mean, maximize)
            val takePos = mPos >= mNeg
            steps += PathStep(layout.predicate(node.split), takePos)
            node = if (takePos) node.pos else node.neg
        }
        return LeafChoice(node as RegressionLeafNode<Vector>, steps)
    }

    private fun signed(m: Double, maximize: Boolean): Double = if (maximize) m else -m

    companion object {
        /**
         * Merge subtree aggregates across trees into one weighted-variance result.
         * Used by [RandomForestBandit] to combine the per-tree frontier statistics for
         * a candidate direction before scoring it.
         */
        fun mergeAggregates(snaps: Iterable<WeightedVarianceResult>): WeightedVarianceResult =
            mergeForestEvidence(snaps, independent = true)
    }
}

/** One step of a root-to-leaf path: which [Split] routed here, and the side taken. */
data class PathStep(val split: Split, val tookPos: Boolean)

/** Walk result: the chosen kumulant leaf node plus the sequence of splits taken to it. */
data class LeafChoice(
    /** Live Kumulant leaf reached by the walk. */
    val leaf: RegressionLeafNode<Vector>,
    val path: List<PathStep>,
)
