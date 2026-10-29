package com.eignex.combo.bo

import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.decisions.context
import com.eignex.combo.shared.LinearFeatureProjection
import com.eignex.combo.shared.WitnessTermination
import com.eignex.kumulant.core.PairedStat
import com.eignex.kumulant.stat.score.MaeLossStat
import com.eignex.kumulant.stat.summary.WeightedMeanResult
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class TrialsTest {
    private class Toggles : DecisionSpace() {
        val a by boolVar()
        val b by boolVar()
        val c by boolVar()
        val premium by contextBool()
    }

    @Test
    fun `batch and subsequent requests exclude all pending assignments without inventing rewards`() {
        val model = Toggles()
        val space = model.compileSpace()
        val ctx = context { set(model.premium, true) }
        for (acquisition in ForestAcquisition.entries) {
            val optimizer = RandomForestOptimizer(
                LinearFeatureProjection(space),
                config = ForestOptimizerConfig(
                    acquisition = acquisition,
                    candidateCount = 8,
                    maxWitnessAttempts = 32,
                    localSteps = 0,
                    initialRandomObservations = 0,
                    randomExploration = 0.0,
                ),
            )
            val warmup = optimizer.suggest(1, ctx)[0]
            optimizer.complete(warmup.id, 1.0)
            val first = optimizer.suggest(3, ctx)
            val second = optimizer.suggest(3, ctx)
            val combined = first.trials + second.trials
            assertEquals(WitnessTermination.CountReached, first.termination)
            assertEquals(6, combined.map { it.sample }.distinct().size)
            assertEquals(6, optimizer.pendingTrials.size)
            assertTrue(combined.all { space.matches(it.sample.sample, space.assumptionsFor(ctx)) })
            assertEquals(1L, optimizer.store.revision)
        }
    }

    @Test
    fun `completion can arrive out of order and records each weighted outcome once`() {
        val model = Toggles()
        val optimizer = RandomForestOptimizer.smac(model.compileSpace())
        val batch = optimizer.suggest(2, context { set(model.premium, false) })
        optimizer.complete(batch[1].id, 7.0, 2.0)
        optimizer.complete(batch[0].id, 3.0)
        val history = optimizer.store.snapshot()
        assertEquals(listOf(7.0, 3.0), history.observations.map { it.reward })
        assertEquals(listOf(2.0, 1.0), history.observations.map { it.weight })
        assertTrue(optimizer.pendingTrials.isEmpty())
        assertTrue(optimizer.trials.all { it.status == TrialStatus.Completed })
        assertEquals(TrialStatus.Pending, batch[0].status)
        assertFailsWith<IllegalArgumentException> { optimizer.complete(batch[1].id, 99.0) }
        assertEquals(2L, optimizer.store.revision)
    }

    @Test
    fun `failed and cancelled trials release reservations without training`() {
        val model = Toggles()
        val optimizer = RandomForestOptimizer.smac(model.compileSpace())
        val ctx = context { set(model.premium, false) }
        val batch = optimizer.suggest(8, ctx)
        assertEquals(8, batch.size)
        val exhausted = optimizer.suggest(1, ctx)
        assertEquals(0, exhausted.size)
        assertEquals(WitnessTermination.BudgetExhausted, exhausted.termination)
        optimizer.fail(batch[0].id, "worker failed")
        optimizer.cancel(batch[1].id)
        val retry = optimizer.suggest(2, ctx)
        assertEquals(setOf(batch[0].sample, batch[1].sample), retry.trials.map { it.sample }.toSet())
        assertEquals(0L, optimizer.store.revision)
        assertFailsWith<IllegalArgumentException> { optimizer.complete(batch[0].id, 1.0) }
    }

    @Test
    fun `partial suggestions retain their reservations`() {
        val model = Toggles()
        val optimizer = RandomForestOptimizer.smac(model.compileSpace())
        val batch = optimizer.suggest(10, context { set(model.premium, true) })
        assertEquals(8, batch.size)
        assertEquals(WitnessTermination.BudgetExhausted, batch.termination)
        assertEquals(8, optimizer.pendingTrials.size)
    }

    @Test
    fun `invalid outcomes leave a reservation pending`() {
        val model = Toggles()
        val optimizer = RandomForestOptimizer.smac(model.compileSpace())
        val trial = optimizer.suggest(1, context { set(model.premium, true) })[0]
        for (weight in listOf(0.0, -1.0, Double.NaN)) {
            assertFailsWith<IllegalArgumentException> { optimizer.complete(trial.id, 1.0, weight) }
        }
        assertFailsWith<IllegalArgumentException> { optimizer.complete(trial.id, Double.NaN) }
        assertFailsWith<IllegalArgumentException> { optimizer.cancel(-1) }
        assertEquals(listOf(trial), optimizer.pendingTrials)
        assertEquals(0L, optimizer.store.revision)
    }

    @Test
    fun `a post recording metric failure cannot append a trial outcome twice`() {
        val model = Toggles()
        val sink = object : PairedStat<WeightedMeanResult> by MaeLossStat() {
            override fun update(x: Double, y: Double, timestampNanos: Long, weight: Double) = error(
                "metric unavailable",
            )
        }
        val optimizer = RandomForestOptimizer(LinearFeatureProjection(model.compileSpace()), trainScore = sink)
        val trial = optimizer.suggest(1, context { set(model.premium, true) })[0]
        assertFailsWith<IllegalStateException> { optimizer.complete(trial.id, 5.0) }
        assertEquals(TrialStatus.Completed, optimizer.trials.single().status)
        assertEquals(1L, optimizer.store.revision)
        assertFailsWith<IllegalArgumentException> { optimizer.complete(trial.id, 5.0) }
        assertEquals(1L, optimizer.store.revision)
    }

    @Test
    fun `state transfer preserves pending reservations and identifiers`() {
        val model = Toggles()
        val space = model.compileSpace()
        val ctx = context { set(model.premium, false) }
        val original = RandomForestOptimizer.smac(space)
        val first = original.suggest(2, ctx)
        original.complete(first[1].id, 9.0)
        val replica = RandomForestOptimizer.bootstrapThompson(space)
        replica.importData(original.exportData())
        assertEquals(original.pendingTrials, replica.pendingTrials)
        val next = replica.suggest(1, ctx)[0]
        assertNotEquals(first[0].sample, next.sample)
        assertTrue(next.id > first[1].id)
        replica.complete(first[0].id, 5.0)
        assertEquals(2L, replica.store.revision)
        assertEquals(1L, original.store.revision)
    }
}
