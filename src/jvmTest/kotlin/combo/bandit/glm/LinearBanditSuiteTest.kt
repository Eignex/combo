package combo.bandit.glm

import com.eignex.klause.solver.LocalSearchParams
import com.eignex.klause.solver.LocalSearchSolver
import com.eignex.klause.solver.Sample
import com.eignex.kumulant.stat.summary.MeanStat
import combo.bandit.Bandit
import combo.bandit.BanditTestSuite
import combo.bandit.PredictionBandit
import combo.bandit.PredictionBanditTestSuite
import combo.decisions.CompiledDecisionSpace

/**
 * Plug [LinearBandit] into the shared bandit harness.
 *
 * Convergence under Bernoulli rewards is harder for the linear model than for a beta-
 * Bernoulli bandit because the model trains a linear regression on 0/1 features rather
 * than tracking sufficient stats per arm. Disable the strict win-rate assertion via
 * [converges]=false; the rest of the contract still applies.
 */
class LinearBanditSuiteTest : PredictionBanditTestSuite<LinearData>() {

    override val converges: Boolean = false

    override fun buildPrediction(
        space: CompiledDecisionSpace,
        samples: List<Sample>,
        randomSeed: Int,
        maximize: Boolean,
        rewards: MeanStat?,
    ): PredictionBandit<LinearData> {
        val projection = LinearFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)
        val model = DiagonalizedLinearModel.Builder(projection.featureSize)
            .family(NormalVariance)
            .learningRate(ConstantRate(1f))
            .priorPrecision(0.01f)
            .exploration(0.2f)
            .build()
        return LinearBandit(
            projection = projection,
            linearModel = model,
            innerOptimizer = { obj, asm ->
                solver.minimize(obj, LocalSearchParams(maxFlips = 200L, randomSeed = randomSeed.toLong(), assumptions = asm))
            },
            randomSeed = randomSeed,
            maximize = maximize,
            rewards = rewards,
        )
    }
}
