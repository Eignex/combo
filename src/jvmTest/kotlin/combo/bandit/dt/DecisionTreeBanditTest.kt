package combo.bandit.dt

import com.eignex.klause.solver.LocalSearchParams
import com.eignex.klause.solver.LocalSearchSolver
import com.eignex.klause.solver.Sample
import com.eignex.kumulant.stat.summary.MeanStat
import combo.bandit.PredictionBandit
import combo.bandit.PredictionBanditTestSuite
import combo.bandit.univariate.Greedy
import combo.decisions.CompiledDecisionSpace

/**
 * Plug [DecisionTreeBandit] into the shared bandit harness. The harness's `TinySpace`
 * has two unconstrained bool variables — both eligible split candidates — so the tree
 * gets to exercise its branch-growth path on the convergence test.
 */
class DecisionTreeBanditTest : PredictionBanditTestSuite<DecisionTreeData>() {

    override fun buildPrediction(
        space: CompiledDecisionSpace,
        samples: List<Sample>,
        randomSeed: Int,
        maximize: Boolean,
        rewards: MeanStat?,
    ): PredictionBandit<DecisionTreeData> {
        val solver = LocalSearchSolver(space.compiled.problem)
        val session = com.eignex.klause.solver.LocalSearchSession(solver)
        // TinySpace has 4 feasible samples and 600 rounds — keep splitting cheap.
        val config = TreeConfig(
            splitPeriod = 5,
            minSamplesSplit = 10.0,
            minSamplesLeaf = 2.0,
            delta = 0.1,
        )
        return DecisionTreeBandit(
            space = space,
            policy = Greedy(),
            proposeSample = { rng, assumptions ->
                session.sample(LocalSearchParams(randomSeed = rng.nextLong(), assumptions = assumptions))
            },
            tree = Tree(Greedy(), defaultSplitCandidates(space), config),
            retryBudget = 32,
            randomSeed = randomSeed,
            maximize = maximize,
            rewards = rewards,
        )
    }
}
