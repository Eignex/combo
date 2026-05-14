package combo.bandit.dt

import com.eignex.kumulant.core.Result
import com.eignex.kumulant.core.SeriesStat
import com.eignex.kumulant.stat.summary.WeightedVarianceResult
import combo.bandit.univariate.BanditPolicy
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Tunables for [Tree]. Shared by [DecisionTreeBandit] and [RandomForestBandit] —
 * both consume the same growth machinery; their wrappers add bandit-level concerns
 * (RNG sequencing, candidate proposal, reward sinks).
 */
data class TreeConfig(
    /** Hoeffding-bound confidence threshold. Lower → splits require more evidence. */
    val delta: Double = 0.05,
    /** Multiplicative decay applied to [delta] per depth — slows growth near leaves. */
    val deltaDecay: Double = 0.9,
    /** If the Hoeffding bound itself shrinks below this, the leaf may split even when
     *  the runner-up is close (the classic VFDT "tie-break" parameter). */
    val tau: Double = 0.05,
    /** Minimum total weighted samples at a leaf before split evaluation. */
    val minSamplesSplit: Double = 30.0,
    /** Minimum weighted samples on each side of a candidate split. */
    val minSamplesLeaf: Double = 5.0,
    /** Audit every Nth observation rather than every update. */
    val splitPeriod: Int = 10,
    /** Hard ceiling on tree depth. */
    val maxDepth: Int = 16,
    /** Hard ceiling on internal + leaf nodes. */
    val maxNodes: Int = 1_024,
    /** Split criterion. */
    val metric: SplitMetric = VarianceReduction,
    /**
     * Breiman-style random-subspace selection: at every audit-leaf birth, draw a
     * fresh random subset of this many candidates from the tree's full pool. `null`
     * disables the trick — every leaf considers every candidate — which is the
     * right default for a *single*-tree [DecisionTreeBandit]. [RandomForestBandit]
     * passes an explicit `mtry` per tree.
     */
    val mtry: Int? = null,
)

/**
 * Online VFDT-style decision tree, parameterised by the per-arm kumulant [Result]
 * type [R]. This is a plain data structure — no RNG, no candidate proposal, no
 * bandit interface — so the same machinery serves both [DecisionTreeBandit] and
 * [RandomForestBandit]. The wrapping bandit owns sample proposal and policy
 * scoring; the tree owns growth bookkeeping.
 *
 * Each leaf carries a kumulant arm produced by [policy]. Audit leaves additionally
 * track pos/neg arm pairs per candidate split, and every [TreeConfig.splitPeriod]
 * observations evaluate them against the Hoeffding bound to decide whether to
 * replace themselves with a [SplitNode].
 *
 * Tree growth requires [WeightedVarianceResult]-shaped arm snapshots (so we can
 * compute variance reductions). Policies whose arms expose a different [Result]
 * still work — the tree just stays a single leaf, and the wrapping bandit
 * degrades to a univariate bandit over the whole feasible set.
 */
class Tree<R : Result>(
    private val policy: BanditPolicy<R>,
    private val splitCandidates: List<Split>,
    private val config: TreeConfig = TreeConfig(),
    randomSeed: Int = 0,
) {
    private val random = kotlin.random.Random(randomSeed)
    private val canGrow: Boolean =
        splitCandidates.isNotEmpty() && policy.createArm().read() is WeightedVarianceResult

    private var nbrNodes: Int = 1
    private var root: Node<R> = newLeaf(depth = 0)

    /** Walk to the leaf [row] resolves to. */
    fun findLeaf(row: TreeRow): LeafNode<R> = root.findLeaf(row)

    /**
     * Descend through every [SplitNode] whose direction is determined by [pinned] (a
     * map keyed by klause bool-variable id). Stop at the deepest reachable node:
     *
     *  - a [LeafNode] — every split on the way down was pinned;
     *  - a [SplitNode] whose split is *not* yet pinned — the frontier the
     *    cross-tree forest algorithm needs to score.
     *
     * Only [BoolSplit]s are pinnable here — [IntThresholdSplit]/[NominalSplit]
     * stop the descent (caller will see them as frontier nodes and may choose to
     * either skip them or handle them via a residual-constraint extension).
     *
     * Exposed as a [Tree] method so callers don't need access to [root].
     */
    fun descendTo(pinned: Map<Int, Boolean>, boolIdByName: Map<String, Int>): Node<R> {
        var node: Node<R> = root
        while (node is SplitNode) {
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

    /** Mean of the leaf [row] resolves to — defined for arm snapshots whose Result
     *  type exposes a scalar mean. */
    fun predict(row: TreeRow): Double = scalarMean(findLeaf(row).arm.read())

    /**
     * Canonical walk-down: at each [SplitNode], score both children's aggregate arms
     * via [policy] (one fresh draw each — this is Thompson sampling per branching
     * decision when [policy] is `ThompsonSampling`), descend into the higher, end
     * at a leaf. Returns the chosen leaf and the path of splits taken to reach it
     * — the path lets the bandit translate the route into klause assumptions or
     * filter samples by route.
     */
    fun chooseLeaf(rng: Random, step: Long, maximize: Boolean): LeafChoice<R> {
        val steps = mutableListOf<PathStep>()
        var node: Node<R> = root
        while (node is SplitNode) {
            val sPos = policy.evaluate(node.pos.arm.read(), step, maximize, rng)
            val sNeg = policy.evaluate(node.neg.arm.read(), step, maximize, rng)
            val takePos = sPos >= sNeg
            steps += PathStep(node.split, takePos)
            node = if (takePos) node.pos else node.neg
        }
        return LeafChoice(node as LeafNode, steps)
    }

    /** Variant of [chooseLeaf] that follows the deterministic mean at each split —
     *  i.e. the "argmax exploit-only" path. Used by `optimalOrThrow`. */
    fun optimalLeaf(maximize: Boolean): LeafChoice<R> {
        val steps = mutableListOf<PathStep>()
        var node: Node<R> = root
        while (node is SplitNode) {
            val mPos = signed(scalarMean(node.pos.arm.read()), maximize)
            val mNeg = signed(scalarMean(node.neg.arm.read()), maximize)
            val takePos = mPos >= mNeg
            steps += PathStep(node.split, takePos)
            node = if (takePos) node.pos else node.neg
        }
        return LeafChoice(node as LeafNode, steps)
    }

    /** Fold an observation into the tree, possibly growing it. */
    fun update(row: TreeRow, reward: Double, weight: Double = 1.0) {
        root = updateNode(root, row, reward, weight, depth = 0)
    }

    private fun signed(m: Double, maximize: Boolean): Double = if (maximize) m else -m

    /**
     * Create a fresh arm of this tree's policy type and merge each of [arms] into it.
     * Used by [RandomForestBandit] to combine subtree statistics across trees.
     */
    fun mergeArms(arms: Iterable<SeriesStat<R>>): R {
        val combined = policy.createArm()
        for (a in arms) combined.merge(a.read())
        return combined.read()
    }

    /**
     * Render the tree as nested `if (split) { ... } else { ... }` text using each
     * split's readable form (`BoolExpr.pretty()` underneath). Leaves show the arm's
     * scalar mean. Debug / inspection helper; not for serialisation.
     */
    fun prettyPrint(indent: String = ""): String = buildString {
        prettyPrintTo(this, root, indent)
    }

    private fun prettyPrintTo(sb: StringBuilder, node: Node<R>, indent: String) {
        when (node) {
            is SplitNode -> {
                sb.append(indent).append("if (").append(node.split.toString()).append(") {\n")
                prettyPrintTo(sb, node.pos, "$indent  ")
                sb.append(indent).append("} else {\n")
                prettyPrintTo(sb, node.neg, "$indent  ")
                sb.append(indent).append("}\n")
            }
            is LeafNode -> {
                val mean = scalarMean(node.arm.read())
                sb.append(indent).append("leaf mean=").append("%.3f".format(mean)).append('\n')
            }
        }
    }

    private fun newLeaf(depth: Int): LeafNode<R> {
        if (depth >= config.maxDepth || nbrNodes + 1 > config.maxNodes || !canGrow) {
            return TerminalLeaf(policy.createArm())
        }
        val subset = pickCandidates()
        return AuditLeaf(
            arm = policy.createArm(),
            candidates = subset,
            pos = List(subset.size) { policy.createArm() },
            neg = List(subset.size) { policy.createArm() },
        )
    }

    /** Breiman-style per-node random subspace: shuffle the full candidate pool and
     *  take the first [TreeConfig.mtry]. `null` (default) returns every candidate. */
    private fun pickCandidates(): List<Split> {
        val k = config.mtry ?: return splitCandidates
        if (k >= splitCandidates.size) return splitCandidates
        return splitCandidates.shuffled(random).take(k)
    }

    private fun updateNode(node: Node<R>, row: TreeRow, reward: Double, weight: Double, depth: Int): Node<R> =
        when (node) {
            is SplitNode -> {
                // Aggregate stat at the split level — feeds the walk-down at choose time.
                policy.update(node.arm, reward, weight)
                if (node.split.direction(row))
                    node.pos = updateNode(node.pos, row, reward, weight, depth + 1)
                else
                    node.neg = updateNode(node.neg, row, reward, weight, depth + 1)
                node
            }
            is TerminalLeaf -> {
                policy.update(node.arm, reward, weight)
                node
            }
            is AuditLeaf -> updateAuditLeaf(node, row, reward, weight, depth)
        }

    private fun updateAuditLeaf(
        leaf: AuditLeaf<R>,
        row: TreeRow,
        reward: Double,
        weight: Double,
        depth: Int,
    ): Node<R> {
        policy.update(leaf.arm, reward, weight)
        for ((i, split) in leaf.candidates.withIndex()) {
            if (split.direction(row)) policy.update(leaf.pos[i], reward, weight)
            else policy.update(leaf.neg[i], reward, weight)
        }
        leaf.observationsSinceLastCheck++
        if (leaf.observationsSinceLastCheck < config.splitPeriod) return leaf
        leaf.observationsSinceLastCheck = 0

        val total = variance(leaf.arm)
        if (total.totalWeights < config.minSamplesSplit) return leaf
        val pos = leaf.pos.map(::variance)
        val neg = leaf.neg.map(::variance)
        val ranked = config.metric.rank(total, pos, neg, config.minSamplesSplit, config.minSamplesLeaf)
        if (ranked.bestIndex < 0 || ranked.top1 <= 0.0) return leaf

        val eps = hoeffdingBound(config.delta, total.totalWeights, depth, config.deltaDecay)
        val passesHoeffding = ranked.top1 - ranked.top2 > eps
        val passesTau = eps < config.tau
        if (!passesHoeffding && !passesTau) return leaf

        nbrNodes += 2
        // The split node inherits the leaf's running aggregate stat: every observation
        // that produced this leaf also passed through this newly-internal node.
        return SplitNode(
            split = leaf.candidates[ranked.bestIndex],
            pos = newLeaf(depth + 1),
            neg = newLeaf(depth + 1),
            arm = leaf.arm,
        )
    }

    private fun variance(arm: SeriesStat<R>): WeightedVarianceResult =
        arm.read() as WeightedVarianceResult

    private fun hoeffdingBound(delta: Double, n: Double, depth: Int, decay: Double): Double {
        if (n <= 0.0) return Double.POSITIVE_INFINITY
        val adjusted = delta * decay.pow(depth.toDouble())
        return sqrt(-ln(adjusted) / (2.0 * n))
    }
}

/** One step of a root-to-leaf path: which [Split] was at this node, and which side
 *  ("pos" / true, "neg" / false) the walk took. */
data class PathStep(val split: Split, val tookPos: Boolean)

/** Result of a walk-down: the chosen leaf plus the sequence of splits taken to it. */
data class LeafChoice<R : com.eignex.kumulant.core.Result>(
    val leaf: LeafNode<R>,
    val path: List<PathStep>,
)

/** Read a scalar mean from any of the kumulant Result types currently used as
 *  bandit-arm sufficient statistics. Used by both DT and RF for prediction. */
internal fun scalarMean(snapshot: Result): Double = when (snapshot) {
    is WeightedVarianceResult -> snapshot.mean
    is com.eignex.kumulant.stat.summary.WeightedMeanResult -> snapshot.mean
    is com.eignex.kumulant.stat.summary.MomentsResult -> snapshot.mean
    is com.eignex.kumulant.stat.summary.BernoulliSumResult ->
        if (snapshot.trials <= 0.0) 0.0 else snapshot.successes / snapshot.trials
    else -> error("Tree can't read mean from snapshot type ${snapshot::class}")
}
