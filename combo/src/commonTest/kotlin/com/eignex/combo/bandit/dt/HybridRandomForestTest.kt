package com.eignex.combo.bandit.dt

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.decisions.context
import com.eignex.combo.shared.DescentSession
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.combo.test.TimedTest
import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.backtrack.BacktrackSolver
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.propagation.bake
import com.eignex.klause.schema.and
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
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class HybridRandomForestTest : TimedTest() {
    private val config = RegressionTreeConfig(
        splitPeriod = 2,
        minSamplesSplit = 6.0,
        minSamplesLeaf = 1.0,
        delta = 0.2,
        tau = 0.1,
    )

    private class ContextSpace : DecisionSpace() {
        val premium by contextBool()
        val age by optionalContextInt(0, 10)
        val a by boolVar()
        val b by boolVar()
        val exclusive by constraint { !(a and b) }
    }

    @Test
    fun `forest learns different actions for fixed context and keeps every witness feasible`() {
        val model = ContextSpace()
        val space = model.compileSpace()
        val solver = BacktrackSolver(space.compiled.problem.bake())
        val forest = RandomForestBandit.build(
            space,
            nbrTrees = 5,
            mtry = 100,
            config = config,
            candidateBatchSize = 4,
            randomSeed = 17,
        )
        val contexts = listOf(
            context { set(model.premium, false) },
            context {
                set(model.premium, true)
                set(model.age, 7)
            },
        )
        val a = space.compiled.boolVarIdByName.getValue(model.a.name)
        val b = space.compiled.boolVarIdByName.getValue(model.b.name)
        val training = contexts.map { ctx ->
            ctx to solver.enumerate(BacktrackParams(assumptions = space.assumptionsFor(ctx))).toList()
        }
        repeat(150) {
            for ((ctx, samples) in training) {
                for (raw in samples) {
                    val reward = if (ctx[model.premium]) raw.boolValue(a) else raw.boolValue(b)
                    forest.update(BanditSample(raw), ctx, if (reward) 1.0 else 0.0)
                }
            }
        }
        for (ctx in contexts) {
            repeat(15) {
                val sample = forest.chooseOrThrow(ctx)
                assertTrue(space.matches(sample.sample, space.assumptionsFor(ctx)))
                assertTrue(!(sample.sample.boolValue(a) && sample.sample.boolValue(b)))
            }
            val best = forest.optimalOrThrow(ctx)
            assertEquals(ctx[model.premium], best.sample.boolValue(a))
            assertEquals(!ctx[model.premium], best.sample.boolValue(b))
        }
        val sample = forest.chooseOrThrow(contexts[0])
        assertFailsWith<IllegalArgumentException> { forest.update(sample, contexts[1], 1.0) }
        assertEquals(0, assertNotNull(forest.descentSession).depth)
        val hybrid = forest.descentSession as HybridDescentSession
        assertTrue(hybrid.poolHits > 0)
    }

    private class NominalSpace : DecisionSpace() {
        val mode by nominal("slow", "fast", "small")
    }

    @Test
    fun `nominal predicates participate in the original cross-tree descent`() {
        val model = NominalSpace()
        val space = model.compileSpace()
        val solver = BacktrackSolver(space.compiled.problem.bake())
        val forest = RandomForestBandit.build(space, nbrTrees = 3, mtry = 100, config = config, randomSeed = 31)
        val samples = solver.enumerate(BacktrackParams()).toList()
        repeat(100) {
            for (raw in samples) {
                forest.update(BanditSample(raw), if (space.compiled.decode(model.mode, raw) == "fast") 1.0 else 0.0)
            }
        }
        repeat(10) {
            assertEquals("fast", space.compiled.decode(model.mode, forest.optimalOrThrow().sample))
        }
    }

    private class FloatSpace : DecisionSpace() {
        val value by floatVar(0.0, 1.0)
    }

    @Test
    fun `native float decisions use scored feasible candidates without a grid`() {
        val model = FloatSpace()
        val space = model.compileSpace()
        val solver = BacktrackSolver(space.compiled.problem.bake())
        val forest = RandomForestBandit.build(space, nbrTrees = 3, mtry = 100, config = config, randomSeed = 43)
        val low = assertNotNull(solver.minimize(space.compiled.minimize(model.value), BacktrackParams()).assignment)
        val high = assertNotNull(solver.minimize(space.compiled.maximize(model.value), BacktrackParams()).assignment)
        repeat(100) {
            forest.update(BanditSample(low), 0.0)
            forest.update(BanditSample(high), 1.0)
        }
        repeat(5) {
            val sample = forest.optimalOrThrow()
            assertTrue(sample.float(model.value, space) in 0.0..1.0)
            assertEquals(forest.predict(BanditSample(high)), forest.predict(sample), 1e-8)
            assertEquals(space.compiled.decode(model.value, sample.sample), sample.float(model.value, space))
        }
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
        repeat(100) {
            forest.update(BanditSample(low), 0.0)
            forest.update(BanditSample(high), 1.0)
        }
        assertEquals(false, forest.optimalOrThrow().sample.boolValue(0))
        assertTrue(hybrid.backtrackCalls > 0)
        assertEquals(0, hybrid.depth)
    }

    @Test
    fun `the hybrid factory can minimize observed costs`() {
        val space = ToggleSpace().compileSpace()
        val forest = RandomForestBandit.build(
            space,
            nbrTrees = 3,
            config = config,
            randomSeed = 23,
            maximize = false,
        )
        repeat(100) {
            forest.update(BanditSample(Sample(booleanArrayOf(false), longArrayOf())), 0.0)
            forest.update(BanditSample(Sample(booleanArrayOf(true), longArrayOf())), 1.0)
        }
        assertEquals(false, forest.optimalOrThrow().sample.boolValue(0))
    }

    @Test
    fun `session scopes are cleaned up even when a backend throws during the base search`() {
        val space = NominalSpace().compileSpace()
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
                solver.minimize(objective, BacktrackParams(assumptions = scope)).assignment
            },
        )
        val ctx = context {
            set(model.premium, true)
            set(model.age, 8)
        }
        val sample = forest.chooseOrThrow(ctx)
        assertEquals(space.assumptionsFor(ctx), fallbackScope)
        assertTrue(space.matches(sample.sample, space.assumptionsFor(ctx)))
    }
}
