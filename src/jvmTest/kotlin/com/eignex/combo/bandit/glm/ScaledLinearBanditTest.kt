package com.eignex.combo.bandit.glm

import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.decisions.Scale
import com.eignex.combo.decisions.context
import com.eignex.combo.shared.LinearFeatureProjection
import com.eignex.kumulant.stat.regression.glm.DiagonalRegressionStat
import com.eignex.kumulant.stat.regression.glm.FactorisedGaussian
import kotlin.test.Test
import kotlin.test.assertTrue

class ScaledLinearBanditTest {
    private class Space : DecisionSpace() {
        val rate by floatVar(1e-6, 1.0, scale = Scale.Log)
        val premium by contextBool()
    }

    @Test
    fun `nonlinear scales use feasible candidate scoring for both posterior and greedy choices`() {
        val model = Space()
        val space = model.compileSpace()
        val projection = LinearFeatureProjection(space)
        val bandit = LinearBandit(
            projection,
            DiagonalRegressionStat(featureSize = projection.featureSize),
            FactorisedGaussian,
            innerOptimizer = { _, _ -> error("a log feature cannot become a linear source objective") },
            randomSeed = 31,
        )
        val ctx = context { set(model.premium, true) }
        val chosen = bandit.chooseOrThrow(ctx)
        bandit.update(chosen, ctx, 1.0)
        val best = bandit.optimalOrThrow(ctx)
        assertTrue(space.matches(chosen.sample, space.assumptionsFor(ctx)))
        assertTrue(space.matches(best.sample, space.assumptionsFor(ctx)))
        assertTrue(best.float(model.rate, space) in 1e-6..1.0)
        assertTrue(bandit.predict(best, ctx).isFinite())
    }
}
