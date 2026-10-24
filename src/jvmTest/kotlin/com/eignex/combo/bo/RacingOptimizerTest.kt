package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.shared.WitnessTermination
import com.eignex.combo.shared.assignmentKey
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.Sample
import com.eignex.kumulant.core.SeriesStat
import com.eignex.kumulant.stat.summary.SumResult
import com.eignex.kumulant.stat.summary.SumStat
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

class RacingOptimizerTest {
    private class Rate : DecisionSpace() {
        val x by floatVar(0.0, 1.0)
    }
    private fun optimizer(
        space: com.eignex.combo.decisions.CompiledDecisionSpace,
        metric: SeriesStat<*>? = null,
    ): BayesianOptimizer {
        val points = listOf(0.0, 0.5, 1.0).map {
            Sample(
                booleanArrayOf(),
                longArrayOf(),
                doubleArrayOf(it),
                exactReals = listOf(BigFraction.ofDouble(it)!!),
            )
        }
        return BayesianOptimizer(
            SearchTransform(space),
            ForestSurrogate(),
            outcome = OutcomeSpec(noise = ObservationNoise.Noisy),
            trainAbsError = metric,
            repeatPolicy = RepeatPolicy.Never,
            initialDesign = InitialDesign { _, _ -> false },
            acquisitionOptimizer = AcquisitionOptimizer { _, request, _ ->
                val excluded = request.excluded.map { it.assignmentKey() }.toSet()
                CandidateSelection(
                    points.firstOrNull { it.assignmentKey() !in excluded }?.let(::BanditSample),
                    WitnessTermination.BudgetExhausted,
                    0,
                )
            },
        )
    }
    private val keys = (1..3).map { EvaluationKey(instance = "dataset", seed = it) }

    @Test
    fun `challengers race on matched keys and every raw measurement trains exactly once`() {
        val model = Rate()
        val space = model.compileSpace()
        val base = optimizer(space)
        val race = RacingOptimizer(base, keys)
        val initial = race.ask(3).tasks
        assertEquals(keys, initial.map { it.key })
        assertEquals(1, initial.map { it.configurationId }.distinct().size)
        race.complete(initial[2].id, 10.0)
        race.complete(initial[0].id, 10.0)
        assertNull(race.incumbent)
        race.complete(initial[1].id, 10.0)
        assertEquals(10.0, race.incumbent!!.reward)
        val good = race.ask(1).tasks.single()
        assertEquals(0.5, good.sample.float(model.x, space))
        race.complete(good.id, 12.0)
        assertEquals(10.0, race.incumbent!!.reward)
        val second = race.ask(1).tasks.single()
        race.complete(second.id, 12.0)
        val third = race.ask(1).tasks.single()
        race.complete(third.id, 12.0)
        assertEquals(12.0, race.incumbent!!.reward)
        val bad = race.ask(1).tasks.single()
        race.complete(bad.id, 8.0)
        assertEquals(RaceStatus.Rejected, race.configurations.single { it.id == bad.configurationId }.status)
        assertEquals(7L, base.store.revision)
        assertEquals(7L, race.store.revision)
        assertEquals(
            listOf(10.0, 10.0, 10.0, 12.0, 12.0, 12.0, 8.0),
            base.store.snapshot().observations.map { it.reward },
        )
        assertFailsWith<IllegalArgumentException> { race.complete(good.id, 99.0) }
    }

    @Test
    fun `pending workers and race stages survive transfer without changing original history`() {
        val space = Rate().compileSpace()
        val first = RacingOptimizer(optimizer(space), keys)
        val tasks = first.ask(3).tasks
        first.complete(tasks[0].id, 2.0)
        val second = RacingOptimizer(optimizer(space), keys)
        second.importData(first.exportData())
        second.complete(tasks[2].id, 4.0)
        second.complete(tasks[1].id, 3.0)
        assertEquals(3.0, second.incumbent!!.reward)
        assertEquals(1L, first.optimizer.store.revision)
        assertEquals(3L, second.optimizer.store.revision)
        assertTrue(second.ask(1).tasks.single().id > tasks.last().id)
    }

    @Test
    fun `failed attempts consume budget and retry keys without fabricating responses`() {
        val race = RacingOptimizer(
            optimizer(Rate().compileSpace()),
            listOf(keys.first()),
            policy = RacingPolicy(retries = 1),
            evaluationBudget = 2,
        )
        val failed = race.ask(1).tasks.single()
        race.fail(failed.id, "worker lost")
        assertEquals(0L, race.optimizer.store.revision)
        val retry = race.ask(1).tasks.single()
        assertEquals(failed.key, retry.key)
        race.complete(retry.id, 2.0)
        assertEquals(1L, race.optimizer.store.revision)
        assertEquals(EvaluationBatchStatus.BudgetExhausted, race.ask(1).status)
        assertEquals(2L, race.store.revision)
    }

    @Test
    fun `cancelled keys may be reissued and invalid outcomes keep workers pending`() {
        val race = RacingOptimizer(optimizer(Rate().compileSpace()), listOf(keys.first()))
        val task = race.ask(1).tasks.single()
        assertFailsWith<IllegalArgumentException> { race.complete(task.id, Double.NaN) }
        assertEquals(EvaluationStatus.Pending, race.tasks.single().status)
        race.cancel(task.id)
        assertEquals(task.key, race.ask(1).tasks.single().key)
        assertEquals(0L, race.optimizer.store.revision)
    }

    @Test
    fun `search exhaustion still reports waiting while issued workers can finish`() {
        val race = RacingOptimizer(optimizer(Rate().compileSpace()), listOf(keys.first()))
        race.complete(race.ask(1).tasks.single().id, 10.0)
        val batch = race.ask(3)
        assertEquals(2, batch.tasks.size)
        assertEquals(EvaluationBatchStatus.Waiting, batch.status)
        batch.tasks.forEach { race.complete(it.id, 9.0) }
        assertEquals(EvaluationBatchStatus.SearchExhausted, race.ask(1).status)
    }

    @Test
    fun `failed evaluation storage cannot duplicate a committed model observation and export recovers it`() {
        val space = Rate().compileSpace()
        val unavailable = object : EvaluationStore by InMemoryEvaluationStore() {
            override fun append(record: EvaluationRecord) = error("storage unavailable")
        }
        val race = RacingOptimizer(optimizer(space), listOf(keys.first()), store = unavailable)
        val task = race.ask(1).tasks.single()
        assertFailsWith<IllegalStateException> { race.complete(task.id, 3.0) }
        assertEquals(1L, race.optimizer.store.revision)
        assertEquals(EvaluationStatus.Completed, race.tasks.single().status)
        assertFailsWith<IllegalArgumentException> { race.complete(task.id, 3.0) }
        val restored = RacingOptimizer(optimizer(space), listOf(keys.first()))
        restored.importData(race.exportData())
        restored.ask(1)
        assertEquals(3.0, restored.incumbent!!.reward)
        assertEquals(1L, restored.store.revision)
        assertEquals(1L, restored.optimizer.store.revision)
    }

    @Test
    fun `metric failures after observation commit still close workers exactly once`() {
        val sink = object : SeriesStat<SumResult> by SumStat() {
            override fun update(value: Double, timestampNanos: Long, weight: Double) = error("metric unavailable")
        }
        val race = RacingOptimizer(optimizer(Rate().compileSpace(), sink), listOf(keys.first()))
        val task = race.ask(1).tasks.single()
        assertFailsWith<IllegalStateException> { race.complete(task.id, 5.0) }
        assertEquals(1L, race.optimizer.store.revision)
        assertEquals(1L, race.store.revision)
        assertEquals(EvaluationStatus.Completed, race.tasks.single().status)
        assertEquals(5.0, race.incumbent!!.reward)
        assertFailsWith<IllegalArgumentException> { race.complete(task.id, 5.0) }
    }

    @Test
    fun `shuffled challenger stages compare the same keys rather than unmatched overall means`() {
        val race = RacingOptimizer(optimizer(Rate().compileSpace()), keys, policy = RacingPolicy(randomSeed = 17))
        val initial = race.ask(3).tasks
        val baseline = keys.associateWith { if (it == keys.first()) 100.0 else 0.0 }
        initial.forEach { race.complete(it.id, baseline.getValue(it.key)) }
        val challenger = mutableListOf<EvaluationTask>()
        repeat(3) {
            val task = race.ask(1).tasks.single()
            challenger += task
            race.complete(task.id, baseline.getValue(task.key) + 1.0)
        }
        assertEquals(keys.toSet(), challenger.map { it.key }.toSet())
        assertTrue(challenger.map { it.key } != keys)
        assertEquals(100.0 / 3.0 + 1.0, race.incumbent!!.reward, 1e-12)
        assertEquals(challenger.first().sample, race.incumbent!!.sample)
    }
}
