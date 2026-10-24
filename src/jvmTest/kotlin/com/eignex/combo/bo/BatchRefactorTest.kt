package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.Sample
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.kumulant.stat.regression.tree.BatchForestConfig
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

    @Test
    fun `one frozen batch fits the base model once without recording pending outcomes`() {
        val space = Rate().compileSpace()
        var fits = 0
        val forest = ForestSurrogate(BatchForestConfig(nbrTrees = 3))
        val optimizer = BayesianOptimizer(
            SearchTransform(space),
            Surrogate { data, random ->
                fits++
                forest.fit(data, random)
            },
            initialDesign = RandomInitialDesign(0, 0.0),
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
            val sizes = mutableListOf<Int>()
            val forest = ForestSurrogate(BatchForestConfig(nbrTrees = 3))
            val optimizer = BayesianOptimizer(
                SearchTransform(space),
                Surrogate { data, random ->
                    sizes += data.features.rows
                    forest.fit(data, random)
                },
                initialDesign = RandomInitialDesign(0, 0.0),
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
            SearchTransform(space),
            surrogate,
            outcome = OutcomeSpec(standardize = true),
            initialDesign = RandomInitialDesign(0, 0.0),
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
        fun optimizer() = RandomForestOptimizer.smac(space, randomSeed = 31)
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
    fun `benchmark records simple regret by evaluation cost for forest EI and random search`() {
        fun evaluate(sample: BanditSample): BenchmarkEvaluation {
            val x = sample.sample.reals[0]
            return BenchmarkEvaluation(-(x - 0.73) * (x - 0.73), cost = 2.0)
        }
        fun forest(): BenchmarkResult {
            val space = Rate().compileSpace()
            val optimizer = RandomForestOptimizer.smac(
                space,
                config = ForestOptimizerConfig(
                    forest = BatchForestConfig(nbrTrees = 5),
                    localSteps = 4,
                ),
                randomSeed = 19,
            )
            return OptimizationBenchmark.run(optimizer, 16, ::evaluate, optimum = 0.0, batchSize = 2)
        }
        val first = forest()
        val repeated = forest()
        val space = Rate().compileSpace()
        val random = StudyRunner(
            Study(space),
            RandomSuggestionPolicy(space, HybridDescentSession.klause(space), seed = 19),
        )
        val baseline = OptimizationBenchmark.run(random, 16, ::evaluate, optimum = 0.0, batchSize = 2)
        assertEquals(first.bestValues.toDoubleArray().toList(), repeated.bestValues.toDoubleArray().toList())
        for (result in listOf(first, baseline)) {
            assertEquals(16, result.bestValues.size)
            assertEquals(32.0, result.cumulativeCost[15])
            assertTrue(result.simpleRegret!![15] <= result.simpleRegret!![0])
            assertTrue(result.simpleRegret!![15] < 0.02)
        }
        println(
            "seed=19, evaluations=16, cost=32, " +
                "forest-EI regret=${first.simpleRegret!![15]}, random regret=${baseline.simpleRegret!![15]}",
        )
    }
}
