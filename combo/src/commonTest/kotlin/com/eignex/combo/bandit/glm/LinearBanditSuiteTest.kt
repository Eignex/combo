package com.eignex.combo.bandit.glm

import com.eignex.combo.bandit.PredictionBanditTestSuite
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.shared.LinearFeatureProjection
import com.eignex.combo.shared.PredictionLearner
import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.backtrack.BacktrackSolver
import com.eignex.klause.propagation.bake
import com.eignex.klause.solver.Sample
import com.eignex.kumulant.stat.regression.glm.ConstantRate
import com.eignex.kumulant.stat.regression.glm.DiagonalRegressionStat
import com.eignex.kumulant.stat.regression.glm.FactorisedGaussian
import com.eignex.kumulant.stat.summary.MeanStat

/** Plug [LinearBandit] into the shared bandit contract checks. */
class LinearBanditSuiteTest : PredictionBanditTestSuite<LinearLearnerData>() {

    override fun buildPrediction(
        space: CompiledDecisionSpace,
        samples: List<Sample>,
        randomSeed: Int,
        maximize: Boolean,
        rewards: MeanStat?,
    ): PredictionLearner<LinearLearnerData> {
        val projection = LinearFeatureProjection(space)
        val solver = BacktrackSolver(space.compiled.problem.bake())
        val regression = DiagonalRegressionStat(
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
                solver.minimize(
                    obj,
                    BacktrackParams(maxDecisions = 16L, randomSeed = randomSeed.toLong(), assumptions = asm),
                ).assignment
            },
            randomSeed = randomSeed,
            maximize = maximize,
            rewards = rewards,
        )
    }
}
