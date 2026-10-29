package com.eignex.combo.bo

import com.eignex.combo.bo.model.BatchForestConfig
import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.shared.WitnessBatch
import com.eignex.combo.shared.WitnessSearch
import com.eignex.combo.shared.WitnessTermination
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.Sample
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class BatchRefactorTest {
    private class Rate : DecisionSpace() {
        val value by floatVar(0.0, 1.0)
    }
    private fun point(value: Double) = BanditSample(
        Sample(
            booleanArrayOf(),
            longArrayOf(),
            doubleArrayOf(value),
            exactReals = listOf(BigFraction.ofDouble(value)!!),
        ),
    )

    private fun search(projection: SearchTransform) = LocalAndRandomSearch(
        projection,
        WitnessSearch { request, _ ->
            val points = (1..6).map { point(it / 8.0).sample }.filter { it !in request.exclude }
            WitnessBatch(points.take(request.count), WitnessTermination.CountReached)
        },
        candidateCount = 6,
        localSteps = 0,
    )

    @Test
    fun `one frozen batch fits the base model once without recording pending outcomes`() {
        val space = Rate().compileSpace()
        val projection = SearchTransform(space)
        var fits = 0
        val forest = ForestSurrogate(BatchForestConfig(nbrTrees = 3))
        val optimizer = BayesianOptimizer(
            projection,
            Surrogate { data, random ->
                fits++
                forest.fit(data, random)
            },
            initialDesign = RandomInitialDesign(0, 0.0),
            acquisitionOptimizer = search(projection),
        )
        optimizer.train(point(0.0), 0.0)
        optimizer.train(point(1.0), 1.0)
        val trials = optimizer.suggest(3)
        assertEquals(3, trials.size)
        assertEquals(1, fits)
        assertEquals(2L, optimizer.store.revision)
        assertEquals(1L, optimizer.lastSuggestion!!.modelFits)
        assertTrue(optimizer.lastSuggestion!!.acquisitionEvaluations > 0)
    }

    @Test
    fun `constant liar and posterior mean fantasies refit temporary data without adding observations`() {
        for (fantasy in listOf(FantasyPolicy.ConstantLiar(9.0), FantasyPolicy.PosteriorMean)) {
            val space = Rate().compileSpace()
            val projection = SearchTransform(space)
            val sizes = mutableListOf<Int>()
            val forest = ForestSurrogate(BatchForestConfig(nbrTrees = 3))
            val optimizer = BayesianOptimizer(
                projection,
                Surrogate { data, random ->
                    sizes += data.features.rows
                    forest.fit(data, random)
                },
                initialDesign = RandomInitialDesign(0, 0.0),
                acquisitionOptimizer = search(projection),
                batchPolicy = SequentialBatchPolicy(fantasy),
            )
            optimizer.train(point(0.0), 0.0)
            optimizer.train(point(1.0), 1.0)
            assertEquals(3, optimizer.suggest(3).size)
            assertEquals(listOf(2, 3, 4), sizes)
            assertEquals(1, optimizer.suggest(1).size)
            assertEquals(5, sizes.last())
            assertEquals(listOf(0.0, 1.0), optimizer.store.snapshot().observations.map { it.reward })
        }
    }

    @Test
    fun `conditionable surrogates receive temporary standardized data without another full fit`() {
        val space = Rate().compileSpace()
        val projection = SearchTransform(space)
        var fits = 0
        val conditions = mutableListOf<Double>()
        val surrogate = Surrogate { _, _ ->
            fits++
            object : ConditionableSurrogate {
                override val uncertaintyKind = UncertaintyKind.LatentPosterior
                override fun predict(features: DenseMatrix) = PredictiveMoments(
                    DenseVector.zero(features.rows),
                    DenseVector.zero(features.rows),
                )
                override fun drawFunction(random: Random) = SampledFunction { DenseVector.zero(it.rows) }
                override fun condition(temporary: TrainingData): FittedSurrogate {
                    conditions += temporary.responses.toDoubleArray().toList()
                    return this
                }
            }
        }
        val optimizer = BayesianOptimizer(
            projection,
            surrogate,
            outcome = OutcomeSpec(standardize = true),
            initialDesign = RandomInitialDesign(0, 0.0),
            acquisitionOptimizer = search(projection),
            batchPolicy = SequentialBatchPolicy(FantasyPolicy.ConstantLiar(250.0)),
        )
        optimizer.train(point(0.0), 100.0)
        optimizer.train(point(1.0), 300.0)
        assertEquals(2, optimizer.suggest(2).size)
        assertEquals(1, fits)
        assertEquals(listOf(0.5), conditions)
        assertEquals(2L, optimizer.store.revision)
    }

    @Test
    fun `exported study state preserves round seeds independently of model cache warmup`() {
        val space = Rate().compileSpace()
        val projection = SearchTransform(space)
        fun optimizer() = BayesianOptimizer(
            projection,
            ForestSurrogate(BatchForestConfig(nbrTrees = 3)),
            acquisitionOptimizer = search(projection),
            initialDesign = RandomInitialDesign(0, 0.0),
            randomSeed = 31,
        )
        val original = optimizer()
        original.train(point(0.0), 0.0)
        original.train(point(1.0), 1.0)
        original.suggest(1)
        val transferred = original.exportData()
        val replica = optimizer()
        replica.importData(transferred)
        replica.predict(point(0.2))
        val left = original.suggest(2)
        val right = replica.suggest(2)
        // Source search caches are deliberately not part of model state; both batches remain valid,
        // and the algorithm round and random-stream identity survive state transfer.
        assertEquals(2, left.size)
        assertEquals(2, right.size)
        assertEquals(original.lastSuggestion!!.round, replica.lastSuggestion!!.round)
        assertEquals(original.lastSuggestion!!.seed, replica.lastSuggestion!!.seed)
        assertEquals(original.lastSuggestion!!.historyRevision, replica.lastSuggestion!!.historyRevision)
    }

    @Test
    fun `benchmark records cost and regret for a fixed evaluation sequence`() {
        val space = Rate().compileSpace()
        val points = listOf(point(0.0), point(0.5), point(1.0))
        var next = 0
        val runner = StudyRunner(
            Study(space),
            SuggestionPolicy { _, count, _ ->
                ProposalBatch(
                    points.drop(next).take(count).also { next += it.size },
                    WitnessTermination.CountReached,
                )
            },
        )
        val result = OptimizationBenchmark.run(runner, 3, {
            BenchmarkEvaluation(it.sample.reals[0], cost = 2.0)
        }, optimum = 1.0)
        assertEquals(listOf(0.0, 0.5, 1.0), result.bestValues.toDoubleArray().toList())
        assertEquals(listOf(2.0, 4.0, 6.0), result.cumulativeCost.toDoubleArray().toList())
        assertEquals(listOf(1.0, 0.5, 0.0), result.simpleRegret!!.toDoubleArray().toList())
    }
}
