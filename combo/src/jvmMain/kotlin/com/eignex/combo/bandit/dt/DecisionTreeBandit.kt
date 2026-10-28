package com.eignex.combo.bandit.dt

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.Context
import com.eignex.combo.shared.LearningFeedback
import com.eignex.combo.shared.NoFeasibleSampleException
import com.eignex.combo.shared.PredictionLearner
import com.eignex.combo.util.RandomSequence
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.solver.Sample
import com.eignex.kumulant.bandit.univariate.BanditPolicy
import com.eignex.kumulant.core.PairedStat
import com.eignex.kumulant.core.SeriesStat
import com.eignex.kumulant.stat.regression.tree.TreeRegressionResult
import com.eignex.kumulant.stat.regression.tree.mergeSnapshot
import com.eignex.kumulant.stat.regression.tree.snapshot
import com.eignex.kumulant.stat.summary.WeightedVarianceResult
import kotlin.random.Random

/**
 * Online decision-tree bandit using the canonical *tree-partitions-arms* algorithm:
 * the [tree] partitions the action space into leaf-shaped arms; each [chooseOrThrow]
 * walks the tree from the root, at each split applying [policy] to score both
 * children's aggregate arms and descending into the higher (one fresh Thompson draw
 * per branching decision when [policy] is `ThompsonSampling`); on reaching a leaf,
 * the bandit materialises any feasible sample whose row routes to that leaf via
 * [proposeSample].
 *
 * [proposeSample] mirrors the [com.eignex.combo.bandit.glm.LinearBandit] precedent: it takes a
 * fresh [Random] and a klause [Assumptions] map, returning `null` if no feasible
 * sample matches. The tree pins everything it can (bool splits, one-vs-rest nominal
 * splits) into the assumptions; int-threshold splits and multi-label nominal splits
 * fall through to rejection sampling controlled by [retryBudget].
 *
 * `optimalOrThrow` performs the same walk-down but follows the deterministic mean at
 * each split instead of a policy draw — the no-exploration counterpart.
 */
class DecisionTreeBandit(
    val space: CompiledDecisionSpace,
    val policy: BanditPolicy<WeightedVarianceResult>,
    val proposeSample: (Random, Assumptions) -> Sample?,
    val tree: Tree = Tree(policy, defaultSplitCandidates(space)),
    val retryBudget: Int = 32,
    override val randomSeed: Int = System.currentTimeMillis().toInt(),
    override val maximize: Boolean = true,
    override val rewards: SeriesStat<*>? = null,
    override val trainScore: PairedStat<*>? = null,
    override val testScore: PairedStat<*>? = null,
) : PredictionLearner<DecisionTreeData> {

    private val randomSequence = RandomSequence(randomSeed)
    private val projection = TreeFeatureProjection(space)
    private var step: Long = 0L

    override fun chooseOrThrow(): BanditSample {
        val rng = randomSequence.next()
        val choice = tree.chooseLeaf(rng, step++, maximize)
        return realise(choice, rng)
    }

    override fun optimalOrThrow(): BanditSample {
        val rng = randomSequence.next()
        val choice = tree.optimalLeaf(maximize)
        return realise(choice, rng)
    }

    private fun realise(choice: LeafChoice, rng: Random): BanditSample {
        val m = materialize(space, choice.path)
        val sample = materializeLeaf(
            rng = rng,
            budget = retryBudget,
            proposeSample = proposeSample,
            assumptions = m.assumptions,
            residual = m.residual,
            encode = { projection.encode(it) },
            routesTo = { tree.findLeaf(it) === choice.leaf },
        )
        return sample ?: throw NoFeasibleSampleException(
            "DecisionTreeBandit could not draw a feasible sample routing to the chosen leaf in $retryBudget attempts",
        )
    }

    private val feedback = LearningFeedback(
        encode = { sample, _ -> projection.encode(sample) },
        mean = tree::predict,
        record = tree::update,
        rewards = rewards,
        trainScore = trainScore,
        testScore = testScore,
    )

    override fun predict(sample: BanditSample): Double = feedback.predict(sample, Context.Empty)

    override fun update(sample: BanditSample, reward: Double, weight: Double) =
        feedback.update(sample, Context.Empty, reward, weight)

    override fun train(sample: BanditSample, reward: Double, weight: Double) =
        feedback.train(sample, Context.Empty, reward, weight)

    override fun importData(data: DecisionTreeData) {
        require(
            tree.layout.matches(data.splits),
        ) { "tree split layouts differ" }
        tree.inner.mergeSnapshot(data.state.root)
    }
    override fun exportData(): DecisionTreeData = DecisionTreeData(
        TreeRegressionResult(tree.inner.rootNode().snapshot()),
        tree.layout.splits.map(tree.layout::predicate),
    )
}
