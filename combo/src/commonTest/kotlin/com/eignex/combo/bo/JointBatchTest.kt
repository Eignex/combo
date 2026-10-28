package com.eignex.combo.bo

import com.eignex.combo.bo.model.GaussianProcessConfig
import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.decisions.context
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.combo.shared.WitnessBatch
import com.eignex.combo.shared.WitnessSearch
import com.eignex.combo.shared.WitnessTermination
import com.eignex.combo.shared.assignmentKey
import com.eignex.combo.test.TimedTest
import com.eignex.klause.schema.not
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.Sample
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class JointBatchTest : TimedTest() {
    private class Rate : DecisionSpace() {
        val x by floatVar(0.0, 1.0)
    }
    private fun point(x: Double) = Sample(
        booleanArrayOf(),
        longArrayOf(),
        doubleArrayOf(x),
        exactReals = listOf(BigFraction.ofDouble(x)!!),
    )
    private val request = AcquisitionRequest(Context.Empty, emptyList(), emptyList())

    @Test
    fun `coordinate exchange scores whole sets and caches permutations`() {
        val projection = SearchTransform(Rate().compileSpace())
        val points = listOf(point(0.0), point(0.5), point(1.0))
        val search = JointPoolSearch(
            projection,
            WitnessSearch { _, _ ->
                WitnessBatch(points, WitnessTermination.CountReached)
            },
            starts = 4,
            passes = 2,
        )
        repeat(10) { seed ->
            var scores = 0
            val selected = search.maximize(
                JointAcquisition { rows ->
                    scores++
                    assertEquals(2, rows.rows)
                    // A complementarity objective: the best two individual means would both be near 1.
                    (rows[0, 0] - rows[1, 0]) * (rows[0, 0] - rows[1, 0])
                },
                request,
                2,
                Random(seed),
            )
            assertEquals(setOf(0.0, 1.0), selected.samples.map { it.sample.reals[0] }.toSet())
            assertEquals(3, selected.evaluations)
            assertEquals(scores, selected.evaluations)
        }
    }

    @Test
    fun `pool validation removes invalid excluded and duplicate assignments and reports exhaustion`() {
        val projection = SearchTransform(Rate().compileSpace())
        val backend = WitnessSearch { _, _ ->
            WitnessBatch(listOf(point(0.0), point(0.5), point(0.5), point(2.0)), WitnessTermination.CountReached)
        }
        val search = JointPoolSearch(projection, backend)
        val filtered = AcquisitionRequest(Context.Empty, listOf(point(0.0)), listOf(point(0.5)))
        val partial = search.maximize(JointAcquisition { 1.0 }, filtered, 2, Random(1))
        assertEquals(listOf(0.5), partial.samples.map { it.sample.reals[0] })
        assertEquals(WitnessTermination.BudgetExhausted, partial.termination)
        val empty = search.maximize(
            JointAcquisition { error("empty pool must not score") },
            AcquisitionRequest(Context.Empty, listOf(point(0.0), point(0.5)), emptyList()),
            2,
            Random(1),
        )
        assertTrue(empty.samples.isEmpty())
        assertEquals(WitnessTermination.BudgetExhausted, empty.termination)
        assertEquals(0, empty.evaluations)
    }

    @Test
    fun `GP joint rounds reserve together reuse one fit and complete out of order without fantasies`() {
        val problem = refinementProblem("noisy")
        val projection = SearchTransform(problem.space)
        for (factory in listOf(JointExpectedImprovement(8), JointNoisyExpectedImprovement(8))) {
            fun optimizer() = BayesianOptimizer(
                projection,
                GaussianProcessSurrogate(GaussianProcessConfig(hyperparameterTrials = 0)),
                batchPolicy = JointBatchPolicy(
                    factory,
                    JointPoolSearch(projection, noisyWitnesses(), candidateCount = 3, starts = 1, passes = 1),
                ),
                initialDesign = RandomInitialDesign(0, 0.0),
                randomSeed = 47,
                outcome = OutcomeSpec(maximize = false, noise = ObservationNoise.Noisy, standardize = true),
            )
            val optimizer = optimizer()
            optimizer.train(continuousPoint(doubleArrayOf(0.2, 0.3)), Context.Empty, 1.0, noiseVariance = 0.01)
            optimizer.train(continuousPoint(doubleArrayOf(0.7, 0.8)), Context.Empty, 2.0, noiseVariance = 0.01)
            val first = optimizer.suggest(2)
            val replica = optimizer().also { it.importData(optimizer.exportData()) }
            val second = optimizer.suggest(2)
            val replay = replica.suggest(2)
            assertEquals(
                second.map { it.sample.sample.assignmentKey() },
                replay.map { it.sample.sample.assignmentKey() },
            )
            assertEquals(4, (first.trials + second.trials).map { it.sample.sample.assignmentKey() }.distinct().size)
            assertEquals(1L, optimizer.fitCount)
            assertEquals(0L, optimizer.lastSuggestion!!.modelFits)
            assertEquals(2L, optimizer.store.revision)
            assertEquals(4, optimizer.pendingTrials.size)
            for (trial in (first.trials + second.trials).reversed()) {
                optimizer.complete(
                    trial.id,
                    1.5,
                    noiseVariance = 0.01,
                )
            }
            assertEquals(6L, optimizer.store.revision)
            assertTrue(optimizer.pendingTrials.isEmpty())
            assertFailsWith<IllegalArgumentException> { optimizer.complete(first[0].id, 9.0) }
            assertEquals(6L, optimizer.store.revision)
        }
    }

    private class Scoped : DecisionSpace() {
        val premium by contextBool()
        val a by boolVar()
        val b by boolVar()
        val fixed by constraint { !b }
    }

    @Test
    fun `joint baselines include only completed and pending points in the requested hard context`() {
        val model = Scoped()
        val space = model.compileSpace()
        val projection = SearchTransform(space)
        val low = context { set(model.premium, false) }
        val high = context { set(model.premium, true) }
        val premiumId = space.compiled.boolVarIdByName.getValue(model.premium.name)
        val actionId = space.compiled.boolVarIdByName.getValue(model.a.name)
        val backend = WitnessSearch { request, _ ->
            val points = listOf(false, true).map { action ->
                val bools = BooleanArray(space.compiled.problem.numBoolVars)
                bools[premiumId] = request.assumptions.bools.getValue(premiumId)
                bools[actionId] = action
                Sample(bools, longArrayOf())
            }.filter { it !in request.exclude }
            WitnessBatch(
                points.take(request.count),
                if (points.size >= request.count) WitnessTermination.CountReached else WitnessTermination.BudgetExhausted,
            )
        }
        var captured: JointAcquisitionContext? = null
        val optimizer = BayesianOptimizer(
            projection,
            ForestSurrogate(),
            initialDesign = RandomInitialDesign(0, 0.0),
            acquisitionOptimizer = LocalAndRandomSearch(projection, backend, candidateCount = 2, localSteps = 0),
            exploration = backend,
            batchPolicy = JointBatchPolicy(
                JointAcquisitionFactory { input, _ ->
                    captured = input
                    JointAcquisition { 0.0 }
                },
                JointPoolSearch(projection, backend, candidateCount = 2, starts = 1, passes = 1),
            ),
            outcome = OutcomeSpec(noise = ObservationNoise.Noisy),
        )
        // No baseline yet: feasible warmup bypasses joint moments even for a marginal-only model.
        val lowPoint = optimizer.suggest(1, low)[0]
        optimizer.complete(lowPoint.id, 1.0)
        val highPoint = optimizer.suggest(1, high)[0]
        optimizer.complete(highPoint.id, 100.0)
        optimizer.suggest(1, high) // Pending in another hard context must not set the threshold.
        optimizer.suggest(1, low)
        val firstCapture = requireNotNull(captured)
        assertEquals(1, firstCapture.baseline.rows)
        assertEquals(0, firstCapture.pending.rows)
        assertEquals(optimizer.predict(lowPoint.sample, low), firstCapture.incumbent)
        val last = optimizer.suggest(2, low)
        assertEquals(1, last.size) // Only two assignments are possible in this context.
        assertEquals(1, requireNotNull(captured).pending.rows)
        assertEquals(WitnessTermination.BudgetExhausted, last.termination)
        assertTrue(last.all { !space.compiled.decode(model.b, it.sample.sample) })
        assertEquals(2L, optimizer.store.revision)
    }

    @Test
    fun `unsupported joint models and invalid batches fail without partial reservations`() {
        val space = Rate().compileSpace()
        val projection = SearchTransform(space)
        val backend = WitnessSearch { _, _ -> WitnessBatch(listOf(point(0.5)), WitnessTermination.CountReached) }
        fun optimizer(policy: BatchPolicy, bootstrap: Boolean = false) = BayesianOptimizer(
            projection,
            ForestSurrogate(),
            initialDesign = RandomInitialDesign(0, 0.0),
            bootstrapThompson = bootstrap,
            batchPolicy = policy,
        ).also { it.train(BanditSample(point(0.0)), 0.0) }
        val joint = JointBatchPolicy(JointExpectedImprovement(), JointPoolSearch(projection, backend))
        for (optimizer in listOf(optimizer(joint), optimizer(joint, bootstrap = true))) {
            assertFailsWith<IllegalArgumentException> { optimizer.suggest(2) }
            assertTrue(optimizer.pendingTrials.isEmpty())
            assertEquals(1L, optimizer.store.revision)
        }
        val invalid = JointBatchPolicy(
            JointAcquisitionFactory { _, _ -> JointAcquisition { 0.0 } },
            JointAcquisitionOptimizer { _, _, _, _ ->
                JointCandidateSelection(
                    listOf(BanditSample(point(0.5)), BanditSample(point(2.0))),
                    WitnessTermination.CountReached,
                    0,
                )
            },
        )
        val optimizer = optimizer(invalid)
        assertFailsWith<IllegalArgumentException> { optimizer.suggest(2) }
        assertTrue(optimizer.pendingTrials.isEmpty())
    }
}
