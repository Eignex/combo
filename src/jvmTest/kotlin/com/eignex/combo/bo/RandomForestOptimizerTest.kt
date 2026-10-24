package com.eignex.combo.bo

import com.eignex.combo.bo.model.BatchForestConfig
import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.decisions.FeatureEncoder
import com.eignex.combo.decisions.context
import com.eignex.combo.shared.LinearFeatureProjection
import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.backtrack.BacktrackSolver
import com.eignex.klause.propagation.bake
import com.eignex.klause.schema.and
import com.eignex.klause.schema.not
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.Sample
import com.eignex.koblas.DenseVector
import com.eignex.koblas.Vector
import com.eignex.kumulant.stat.summary.SumStat
import org.junit.Rule
import org.junit.rules.Timeout
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class RandomForestOptimizerTest {
    @get:Rule
    val timeout: Timeout = Timeout.seconds(30)

    private class Toggle : DecisionSpace() {
        val flag by boolVar()
    }

    private class Contextual : DecisionSpace() {
        val premium by contextBool()
        val a by boolVar()
        val b by boolVar()
        val exclusive by constraint { !(a and b) }
    }

    @Test
    fun `full history refits preserve observation weights and reuse an unchanged model`() {
        val space = Toggle().compileSpace()
        val store = InMemoryObservationStore()
        val optimizer = RandomForestOptimizer.smac(
            space,
            store,
            ForestOptimizerConfig(forest = BatchForestConfig(maxDepth = 0, bootstrap = false)),
        )
        val sample = BanditSample(Sample(booleanArrayOf(false), longArrayOf()))
        optimizer.train(sample, 1.0)
        optimizer.train(sample, 9.0, weight = 3.0)
        assertEquals(7.0, optimizer.predict(sample), 1e-12)
        assertEquals(7.0, optimizer.predict(sample), 1e-12)
        assertEquals(1L, optimizer.fitCount)
        store.append(Observation(sample, Context.Empty, 2.0))
        assertEquals(6.0, optimizer.predict(sample), 1e-12)
        assertEquals(2L, optimizer.fitCount)
    }

    @Test
    fun `refitting snapshots each row from an encoder that reuses vector storage`() {
        val compiledSpace = Toggle().compileSpace()
        val buffer = DenseVector.zero(1)
        val projection = object : FeatureEncoder<Vector> {
            override val space = compiledSpace
            override val featureSize = 1
            override fun encode(sample: BanditSample, context: Context): Vector {
                buffer[0] = if (sample.sample.boolValue(0)) 1.0 else 0.0
                return buffer
            }
        }
        val optimizer = RandomForestOptimizer(
            projection,
            config = ForestOptimizerConfig(forest = BatchForestConfig(nbrTrees = 1, bootstrap = false)),
        )
        val low = BanditSample(Sample(booleanArrayOf(false), longArrayOf()))
        val high = BanditSample(Sample(booleanArrayOf(true), longArrayOf()))
        optimizer.train(low, 0.0)
        optimizer.train(high, 1.0)
        assertEquals(0.0, optimizer.predict(low))
        assertEquals(1.0, optimizer.predict(high))
    }

    @Test
    fun `bootstrap Thompson sampling rebuilds a complete forest for every choice`() {
        val space = Toggle().compileSpace()
        val optimizer = RandomForestOptimizer.bootstrapThompson(
            space,
            config = ForestOptimizerConfig(
                initialRandomObservations = 0,
                randomExploration = 0.0,
                repeatPolicy = RepeatPolicy.Allow,
            ),
        )
        optimizer.train(BanditSample(Sample(booleanArrayOf(false), longArrayOf())), 0.0)
        optimizer.train(BanditSample(Sample(booleanArrayOf(true), longArrayOf())), 1.0)
        optimizer.chooseOrThrow()
        assertEquals(1L, optimizer.fitCount)
        optimizer.chooseOrThrow()
        assertEquals(2L, optimizer.fitCount)
    }

    @Test
    fun `both acquisition rules learn and preserve hard contexts and constraints`() {
        val model = Contextual()
        val space = model.compileSpace()
        val solver = BacktrackSolver(space.compiled.problem.bake())
        val contexts = listOf(context { set(model.premium, false) }, context { set(model.premium, true) })
        val a = space.compiled.boolVarIdByName.getValue(model.a.name)
        val b = space.compiled.boolVarIdByName.getValue(model.b.name)
        for (acquisition in ForestAcquisition.entries) {
            val optimizer = RandomForestOptimizer(
                LinearFeatureProjection(space),
                config = ForestOptimizerConfig(
                    forest = BatchForestConfig(nbrTrees = 7, mtry = 3),
                    acquisition = acquisition,
                    initialRandomObservations = 0,
                    randomExploration = 0.0,
                    repeatPolicy = RepeatPolicy.Allow,
                ),
                randomSeed = 11,
            )
            for (ctx in contexts) {
                val samples = solver.enumerate(BacktrackParams(assumptions = space.assumptionsFor(ctx))).toList()
                repeat(15) {
                    for (sample in samples) {
                        val wanted = sample.boolValue(if (ctx[model.premium]) a else b)
                        optimizer.train(BanditSample(sample), ctx, if (wanted) 1.0 else 0.0)
                    }
                }
            }
            for (ctx in contexts) {
                val best = optimizer.optimalOrThrow(ctx)
                assertEquals(ctx[model.premium], best.sample.boolValue(a))
                assertEquals(!ctx[model.premium], best.sample.boolValue(b))
                repeat(4) {
                    val chosen = optimizer.chooseOrThrow(ctx)
                    assertTrue(space.matches(chosen.sample, space.assumptionsFor(ctx)))
                    assertTrue(!(chosen.sample.boolValue(a) && chosen.sample.boolValue(b)))
                }
            }
            val wrong = optimizer.optimalOrThrow(contexts[0])
            assertFailsWith<IllegalArgumentException> { optimizer.update(wrong, contexts[1], 1.0) }
        }
    }

    @Test
    fun `expected improvement handles a context with no evaluated configuration`() {
        val model = Contextual()
        val space = model.compileSpace()
        val solver = BacktrackSolver(space.compiled.problem.bake())
        val trained = context { set(model.premium, false) }
        val unseen = context { set(model.premium, true) }
        val optimizer = RandomForestOptimizer.smac(
            space,
            config = ForestOptimizerConfig(initialRandomObservations = 0, randomExploration = 0.0),
        )
        for (sample in solver.enumerate(BacktrackParams(assumptions = space.assumptionsFor(trained)))) {
            optimizer.train(BanditSample(sample), trained, 1.0)
        }
        val chosen = optimizer.chooseOrThrow(unseen)
        assertTrue(space.matches(chosen.sample, space.assumptionsFor(unseen)))
    }

    @Test
    fun `both acquisition rules learn from the choose observe loop`() {
        val space = Toggle().compileSpace()
        for (acquisition in ForestAcquisition.entries) {
            val optimizer = RandomForestOptimizer(
                LinearFeatureProjection(space),
                config = ForestOptimizerConfig(acquisition = acquisition, repeatPolicy = RepeatPolicy.Allow),
                randomSeed = 31,
            )
            repeat(25) {
                val sample = optimizer.chooseOrThrow()
                optimizer.update(sample, if (sample.sample.boolValue(0)) 1.0 else 0.0)
            }
            assertTrue(optimizer.optimalOrThrow().sample.boolValue(0))
            assertEquals(25, optimizer.store.snapshot().observations.size)
        }
    }

    private class Continuous : DecisionSpace() {
        val value by floatVar(0.0, 1.0)
    }

    @Test
    fun `continuous decisions retain native Klause values`() {
        val model = Continuous()
        val space = model.compileSpace()
        val solver = BacktrackSolver(space.compiled.problem.bake())
        val low = BanditSample(
            solver.minimize(space.compiled.minimize(model.value), BacktrackParams(maxDecisions = 100L)).assignment!!,
        )
        val high = BanditSample(
            solver.minimize(space.compiled.maximize(model.value), BacktrackParams(maxDecisions = 100L)).assignment!!,
        )
        for (acquisition in ForestAcquisition.entries) {
            val optimizer = RandomForestOptimizer(
                LinearFeatureProjection(space),
                config = ForestOptimizerConfig(acquisition = acquisition),
                randomSeed = 13,
            )
            repeat(15) {
                optimizer.train(low, 0.0)
                optimizer.train(high, 1.0)
            }
            val best = optimizer.optimalOrThrow()
            assertTrue(best.float(model.value, space) in 0.0..1.0)
            assertEquals(1.0, optimizer.predict(best), 1e-8)
            assertEquals(space.compiled.decode(model.value, best.sample), best.float(model.value, space))
        }
    }

    @Test
    fun `recommendations include feasible interior observations and reject invalid history`() {
        val model = Continuous()
        val space = model.compileSpace()
        fun point(value: Double) = BanditSample(
            Sample(
                booleanArrayOf(),
                longArrayOf(),
                doubleArrayOf(value),
                exactReals = listOf(BigFraction.ofDouble(value)!!),
            ),
        )
        val optimizer = RandomForestOptimizer.smac(
            space,
            config = ForestOptimizerConfig(forest = BatchForestConfig(nbrTrees = 1, bootstrap = false)),
        )
        optimizer.train(point(0.0), 0.0)
        optimizer.train(point(1.0), 0.0)
        optimizer.train(point(0.5), 9.0)
        optimizer.train(point(2.0), 100.0)
        val recommended = optimizer.optimalOrThrow()
        assertTrue(recommended.float(model.value, space) in 0.0..1.0)
        assertEquals(9.0, optimizer.predict(recommended))
    }

    @Test
    fun `minimization recommends the lowest predicted cost`() {
        val space = Toggle().compileSpace()
        val optimizer = RandomForestOptimizer.smac(space, maximize = false)
        repeat(15) {
            optimizer.train(BanditSample(Sample(booleanArrayOf(false), longArrayOf())), 9.0)
            optimizer.train(BanditSample(Sample(booleanArrayOf(true), longArrayOf())), 1.0)
        }
        assertTrue(optimizer.optimalOrThrow().sample.boolValue(0))
    }

    @Test
    fun `history import transfers evaluations independently of later appends`() {
        val space = Toggle().compileSpace()
        val original = RandomForestOptimizer.smac(space)
        val sample = BanditSample(Sample(booleanArrayOf(false), longArrayOf()))
        original.train(sample, 7.0)
        val data = original.exportData()
        original.train(sample, 100.0)
        val replica = RandomForestOptimizer.bootstrapThompson(space)
        replica.importData(data)
        assertEquals(1, replica.store.snapshot().observations.size)
        assertEquals(7.0, replica.predict(sample))
    }

    @Test
    fun `history learners share weighted reward and pre and post training error tracking`() {
        val space = Toggle().compileSpace()
        val rewards = SumStat()
        val before = SumStat()
        val after = SumStat()
        val optimizer = RandomForestOptimizer(
            LinearFeatureProjection(space),
            rewards = rewards,
            testAbsError = before,
            trainAbsError = after,
        )
        val sample = BanditSample(Sample(booleanArrayOf(false), longArrayOf()))
        optimizer.train(sample, 100.0, weight = 0.0)
        assertEquals(0L, optimizer.store.revision)
        assertFailsWith<IllegalArgumentException> { optimizer.train(sample, 1.0, weight = -1.0) }
        optimizer.update(sample, 3.0, weight = 2.0)
        assertEquals(6.0, rewards.read().sum)
        assertEquals(6.0, before.read().sum)
        assertEquals(0.0, after.read().sum)
    }
}
