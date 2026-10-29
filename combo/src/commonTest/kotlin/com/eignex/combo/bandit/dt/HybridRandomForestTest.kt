package com.eignex.combo.bandit.dt

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.decisions.context
import com.eignex.combo.shared.DescentSession
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.backtrack.BacktrackSolver
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.propagation.bake
import com.eignex.klause.schema.not
import com.eignex.klause.solver.Sample
import com.eignex.klause.solver.result.SampleResult
import com.eignex.klause.solver.result.TerminationReason
import com.eignex.kumulant.bandit.univariate.Greedy
import com.eignex.kumulant.stat.regression.tree.RegressionTreeConfig
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class HybridRandomForestTest {
    private val config = RegressionTreeConfig(
        splitPeriod = 2,
        minSamplesSplit = 6.0,
        minSamplesLeaf = 1.0,
        delta = 0.2,
        tau = 0.1,
    )

    private class ContextSpace : DecisionSpace() {
        val premium by contextBool()
        val choice by boolVar()
    }

    private class ToggleSpace : DecisionSpace() {
        val flag by boolVar()
    }

    @Test
    fun `an unresolved preferred branch keeps the previous feasible witness`() {
        val space = ToggleSpace().compileSpace()
        val low = Sample(booleanArrayOf(false), longArrayOf())
        val high = Sample(booleanArrayOf(true), longArrayOf())
        val hybrid = HybridDescentSession(
            space,
            propose = { _, assumptions, _ ->
                if (assumptions.boolValueOrNull(0) == true) {
                    SampleResult.Unknown(TerminationReason.BudgetExhausted)
                } else {
                    SampleResult.Found(low)
                }
            },
            backtrack = { _, _ -> SampleResult.Unknown(TerminationReason.BudgetExhausted) },
            candidateBatchSize = 1,
        )
        val forest = RandomForestBandit.build(
            space,
            policy = Greedy(),
            proposeSample = { rng, scope -> hybrid.search(rng, scope).assignment },
            descentSession = hybrid,
            nbrTrees = 3,
            mtry = 100,
            config = config,
            randomSeed = 19,
        )
        repeat(12) {
            forest.update(BanditSample(low), 0.0)
            forest.update(BanditSample(high), 1.0)
        }
        assertEquals(false, forest.optimalOrThrow().sample.boolValue(0))
        assertTrue(hybrid.backtrackCalls > 0)
        assertEquals(0, hybrid.depth)
    }

    @Test
    fun `session scopes are cleaned up even when a backend throws during the base search`() {
        val space = ToggleSpace().compileSpace()
        val session = object : DescentSession {
            override var depth = 0
            override fun push(assumptions: Assumptions) {
                depth++
            }
            override fun pop() {
                depth--
            }
            override fun sample(rng: Random): Sample? = error("backend failed")
        }
        val forest = RandomForestBandit.build(
            space,
            policy = Greedy(),
            proposeSample = { _, _ -> error("unexpected fallback") },
            descentSession = session,
            nbrTrees = 1,
        )
        assertFailsWith<IllegalStateException> { forest.chooseOrThrow() }
        assertEquals(0, session.depth)
    }

    @Test
    fun `fallback keeps context hard even after greedy decisions cannot be materialized`() {
        val model = ContextSpace()
        val space = model.compileSpace()
        val solver = BacktrackSolver(space.compiled.problem.bake())
        var fallbackScope: Assumptions? = null
        val forest = RandomForestBandit.build(
            space,
            policy = Greedy(),
            proposeSample = { _, _ -> null },
            nbrTrees = 1,
            optimizeFallback = { objective, scope ->
                fallbackScope = scope
                solver.minimize(objective, BacktrackParams(maxDecisions = 32, assumptions = scope)).assignment
            },
        )
        val ctx = context {
            set(model.premium, true)
        }
        val sample = forest.chooseOrThrow(ctx)
        assertEquals(space.assumptionsFor(ctx), fallbackScope)
        assertTrue(space.matches(sample.sample, space.assumptionsFor(ctx)))
    }
}
