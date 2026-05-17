package combo.bandit.glm

import com.eignex.klause.solver.localsearch.LocalSearchParams
import com.eignex.klause.solver.localsearch.LocalSearchSolver
import com.eignex.klause.solver.Sample
import com.eignex.kumulant.bandit.FactorisedGaussian
import com.eignex.kumulant.stat.regression.ConstantRate
import com.eignex.kumulant.stat.regression.DiagonalRegression
import com.eignex.kumulant.stat.summary.MeanStat
import combo.bandit.PredictionLearner
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
class LinearBanditSuiteTest : PredictionBanditTestSuite<LinearLearnerData>() {

    override val converges: Boolean = false

    override fun buildPrediction(
        space: CompiledDecisionSpace,
        samples: List<Sample>,
        randomSeed: Int,
        maximize: Boolean,
        rewards: MeanStat?,
    ): PredictionLearner<LinearLearnerData> {
        val projection = LinearFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)
        val regression = DiagonalRegression(
            featureSize = projection.featureSize,
            priorPrecision = 0.01,
            learningRate = ConstantRate(1.0),
        )
        return LinearBandit(
            projection = projection,
            regression = regression,
            posterior = FactorisedGaussian,
            exploration = 0.2,
            innerOptimizer = { obj, asm ->
                solver.minimize(obj, LocalSearchParams(maxFlips = 200L, randomSeed = randomSeed.toLong(), assumptions = asm))
            },
            randomSeed = randomSeed,
            maximize = maximize,
            rewards = rewards,
        )
    }
}
