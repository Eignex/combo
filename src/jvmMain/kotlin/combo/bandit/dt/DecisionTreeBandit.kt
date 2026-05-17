package combo.bandit.dt

import com.eignex.klause.solver.Assumptions
import com.eignex.klause.solver.Sample
import com.eignex.kumulant.core.Result
import com.eignex.kumulant.core.SeriesStat
import combo.bandit.NoFeasibleSampleException
import combo.bandit.PredictionLearner
import com.eignex.kumulant.bandit.BanditPolicy
import combo.decisions.BanditSample
import combo.decisions.CompiledDecisionSpace
import combo.util.RandomSequence
import kotlin.math.abs
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
 * [proposeSample] mirrors the [combo.bandit.glm.LinearBandit] precedent: it takes a
 * fresh [Random] and a klause [Assumptions] map, returning `null` if no feasible
 * sample matches. The tree pins everything it can (bool splits, one-vs-rest nominal
 * splits) into the assumptions; int-threshold splits and multi-label nominal splits
 * fall through to rejection sampling controlled by [retryBudget].
 *
 * `optimalOrThrow` performs the same walk-down but follows the deterministic mean at
 * each split instead of a policy draw — the no-exploration counterpart.
 */
class DecisionTreeBandit<R : Result>(
    val space: CompiledDecisionSpace,
    val policy: BanditPolicy<R>,
    val proposeSample: (Random, Assumptions) -> Sample?,
    val tree: Tree<R> = Tree(policy, defaultSplitCandidates(space)),
    val retryBudget: Int = 32,
    override val randomSeed: Int = System.currentTimeMillis().toInt(),
    override val maximize: Boolean = true,
    override val rewards: SeriesStat<*>? = null,
    override val trainAbsError: SeriesStat<*>? = null,
    override val testAbsError: SeriesStat<*>? = null,
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

    private fun realise(choice: LeafChoice<R>, rng: Random): BanditSample {
        val m = materialize(space, choice.path)
        val sample = materializeLeaf(
            rng = rng,
            budget = retryBudget,
            space = space,
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

    override fun predict(sample: BanditSample): Double = tree.predict(projection.encode(sample))

    override fun update(sample: BanditSample, reward: Double, weight: Double) {
        rewards?.update(reward, 0L, weight)
        if (testAbsError != null) testAbsError.update(abs(reward - predict(sample)), 0L, weight)
        train(sample, reward, weight)
        if (trainAbsError != null) trainAbsError.update(abs(reward - predict(sample)), 0L, weight)
    }

    override fun train(sample: BanditSample, reward: Double, weight: Double) {
        tree.update(projection.encode(sample), reward, weight)
    }

    // Slice 1: serialisable tree state is a follow-up.
    override fun importData(data: DecisionTreeData) {}
    override fun exportData(): DecisionTreeData = DecisionTreeData()
}
