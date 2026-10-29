package com.eignex.combo.bandit.glm

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.decisions.SubSpace
import com.eignex.combo.decisions.context
import com.eignex.combo.shared.LinearFeatureProjection
import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.backtrack.BacktrackSolver
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.propagation.bake
import com.eignex.kumulant.stat.regression.glm.DiagonalRegressionStat
import com.eignex.kumulant.stat.regression.glm.FactorisedGaussian
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class LinearBanditTest {
    private class Nominal : DecisionSpace() {
        val type by nominal("a", "b", "c")
    }

    private class Contextual : DecisionSpace() {
        val choice by boolVar()
        val premium by contextBool()
        val effect by interact(premium, choice)
    }

    private class Audio : SubSpace() {
        val mute by boolVar()
    }

    private class Optional : DecisionSpace() {
        val audio by optionalDecisionSpace(::Audio)
    }

    @Test
    fun `nominal rewards update predictions and greedy choice selects the best label`() {
        val model = Nominal()
        val space = model.compileSpace()
        val projection = LinearFeatureProjection(space)
        val solver = BacktrackSolver(space.compiled.problem.bake())
        val params = BacktrackParams(maxDecisions = 32, randomSeed = 3)
        val bandit = LinearBandit(
            projection,
            DiagonalRegressionStat(featureSize = projection.featureSize),
            FactorisedGaussian,
            exploration = 0.1,
            innerOptimizer = { objective, assumptions ->
                solver.minimize(objective, params.copy(assumptions = assumptions)).assignment
            },
            randomSeed = 3,
        )
        assertEquals(3, projection.featureSize)
        val samples = solver.enumerate(params).map(::BanditSample).toList()
        assertEquals(3, samples.size)
        for (sample in samples) {
            val reward = if (space.compiled.decode(model.type, sample.sample) == "b") 1.0 else -1.0
            bandit.train(sample, reward)
        }
        val best = bandit.optimalOrThrow()
        assertEquals("b", space.compiled.decode(model.type, best.sample))
        assertTrue(bandit.predict(best) > bandit.predict(samples.first { it != best }))
        assertTrue(bandit.chooseOrThrow().sample in samples.map { it.sample })
    }

    @Test
    fun `context pins choices and activates its interaction feature`() {
        val model = Contextual()
        val space = model.compileSpace()
        val projection = LinearFeatureProjection(space)
        val solver = BacktrackSolver(space.compiled.problem.bake())
        val params = BacktrackParams(maxDecisions = 32, randomSeed = 7)
        val bandit = LinearBandit(
            projection,
            DiagonalRegressionStat(featureSize = projection.featureSize),
            FactorisedGaussian,
            innerOptimizer = { objective, assumptions ->
                solver.minimize(objective, params.copy(assumptions = assumptions)).assignment
            },
            randomSeed = 7,
        )
        for (premium in listOf(false, true)) {
            val ctx = context { set(model.premium, premium) }
            val assumptions = space.assumptionsFor(ctx)
            val samples = solver.enumerate(params.copy(assumptions = assumptions)).toList()
            assertEquals(2, samples.size)
            for (raw in samples) {
                val sample = BanditSample(raw)
                val choice = space.compiled.decode(model.choice, raw)
                assertEquals(
                    if (premium && choice) 1.0 else 0.0,
                    projection.encode(sample, ctx)[projection.layout.interactionsStart],
                )
                bandit.train(sample, ctx, if (choice) 1.0 else -1.0)
            }
            assertTrue(space.matches(bandit.chooseOrThrow(ctx).sample, assumptions))
            assertTrue(space.matches(bandit.optimalOrThrow(ctx).sample, assumptions))
        }
        assertFailsWith<IllegalStateException> { bandit.chooseOrThrow(context {}) }
    }

    @Test
    fun `inactive optional slots are masked in encoding`() {
        val model = Optional()
        val space = model.compileSpace()
        val projection = LinearFeatureProjection(space)
        val solver = BacktrackSolver(space.compiled.problem.bake())
        val gate = space.compiled.boolVarIdByName.getValue(space.gateOf(model.audio)!!.name)
        val mute = space.compiled.boolVarIdByName.getValue("audio.mute")
        for (active in listOf(false, true)) {
            val raw = solver.sample(
                BacktrackParams(
                    maxDecisions = 32,
                    assumptions = Assumptions(bools = mapOf(gate to active, mute to active)),
                ),
            ).assignment!!
            assertEquals(if (active) 1.0 else 0.0, projection.encode(BanditSample(raw))[mute])
        }
    }
}
