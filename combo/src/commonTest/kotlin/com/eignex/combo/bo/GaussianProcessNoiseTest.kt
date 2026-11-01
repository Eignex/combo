package com.eignex.combo.bo

import com.eignex.combo.bo.model.GaussianProcessConfig
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.shared.WitnessBatch
import com.eignex.combo.shared.WitnessSearch
import com.eignex.combo.shared.WitnessTermination
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import kotlin.math.exp
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class GaussianProcessNoiseTest {
    private fun rows(vararg values: Double) = DenseMatrix.ofRows(values.map { doubleArrayOf(it) }.toTypedArray())
    private fun data(features: DenseMatrix, responses: DoubleArray, noise: DoubleArray) = TrainingData(
        features,
        DenseVector.of(responses),
        DenseVector.of(DoubleArray(responses.size) { 1.0 }),
        DenseVector.of(noise),
    )

    @Test
    fun `mixed noise fits and conditions like explicit row variances with the learned fallback`() {
        val config = GaussianProcessConfig(mean = 0.0, learnNoise = true, hyperparameterTrials = 12)
        val training = data(rows(0.0, 0.4, 0.8), doubleArrayOf(1.0, -1.0, 2.0), doubleArrayOf(0.0, Double.NaN, 0.25))
        val fitted = GaussianProcessSurrogate(config).fit(training, Random(3))
        val parameters = fitted.parameters
        val explicitConfig = config.copy(
            lengthScales = parameters.lengthScales,
            signalVariance = parameters.signalVariance,
            mean = parameters.mean,
            noiseVariance = 100.0,
            learnNoise = false,
            hyperparameterTrials = 0,
        )
        val temporary = data(rows(0.2, 0.6), doubleArrayOf(3.0, -2.0), doubleArrayOf(Double.NaN, 0.5))
        val conditioned = fitted.condition(temporary)
        val explicit = GaussianProcessSurrogate(explicitConfig).fit(
            data(
                rows(0.0, 0.4, 0.8, 0.2, 0.6),
                doubleArrayOf(1.0, -1.0, 2.0, 3.0, -2.0),
                doubleArrayOf(0.0, parameters.noiseVariance, 0.25, parameters.noiseVariance, 0.5),
            ),
            Random(3),
        )
        val queries = rows(0.1, 0.5, 0.9)
        val expected = explicit.predictJoint(queries)
        val actual = conditioned.predictJoint(queries)
        for (i in 0 until queries.rows) {
            assertEquals(expected.means[i], actual.means[i], 1e-12)
            for (j in 0 until queries.rows) assertEquals(expected.covariance[i, j], actual.covariance[i, j], 1e-12)
        }
        assertEquals(parameters, conditioned.parameters)
        assertEquals(0.0, training.noiseVariances!![0])
        assertTrue(training.noiseVariances[1].isNaN())
        assertEquals(0.25, training.noiseVariances[2])
        assertTrue(temporary.noiseVariances!![0].isNaN())
    }

    private class Rate : DecisionSpace() {
        val x by floatVar(0.0, 1.0)
    }

    @Test
    fun `mixed noise standardization preserves affine response units through fitting and conditioning`() {
        val space = Rate().compileSpace()
        val projection = SearchTransform(space)
        val base = listOf(2.0, 4.0, 8.0)
        val noises = listOf(0.0, 0.25, null)
        fun fit(center: Double, scale: Double): Pair<FittedSurrogate, TrainingData> {
            var captured: TrainingData? = null
            val learning = SurrogateLearning(
                projection,
                InMemoryObservationStore(),
                Surrogate { training, random ->
                    captured = training
                    GaussianProcessSurrogate(GaussianProcessConfig(learnNoise = true, hyperparameterTrials = 0))
                        .fit(training, random)
                },
                OutcomeSpec(standardize = true, noise = ObservationNoise.Noisy),
                7,
                null,
                null,
                null,
            )
            val model = learning.fit(
                base.mapIndexed { i, response ->
                    Observation(
                        continuousPoint(doubleArrayOf(i * 0.4)),
                        Context.Empty,
                        center + scale * response,
                        noiseVariance = noises[i]?.let { it * scale * scale },
                    )
                },
                Random(7),
            )
            return model to assertNotNull(captured)
        }
        val (baseModel, baseTraining) = fit(0.0, 1.0)
        val (scaledModel, scaledTraining) = fit(10.0, 3.0)
        for (i in base.indices) {
            assertEquals(baseTraining.responses[i], scaledTraining.responses[i], 1e-12)
            assertEquals(1.0, scaledTraining.weights[i])
            if (noises[i] == null) {
                assertTrue(baseTraining.noiseVariances!![i].isNaN())
                assertTrue(scaledTraining.noiseVariances!![i].isNaN())
            } else {
                assertEquals(baseTraining.noiseVariances!![i], scaledTraining.noiseVariances!![i], 1e-12)
            }
        }
        val temporary = data(rows(0.2, 0.6), doubleArrayOf(3.0, 5.0), doubleArrayOf(0.5, Double.NaN))
        val scaledTemporary = data(rows(0.2, 0.6), doubleArrayOf(19.0, 25.0), doubleArrayOf(4.5, Double.NaN))
        val conditioned = (baseModel as ConditionableSurrogate).condition(temporary)
        val scaledConditioned = (scaledModel as ConditionableSurrogate).condition(scaledTemporary)
        val queries = rows(0.1, 0.5, 0.9)
        for ((left, right) in listOf(baseModel to scaledModel, conditioned to scaledConditioned)) {
            val expected = assertNotNull(left.predictJoint(queries))
            val actual = assertNotNull(right.predictJoint(queries))
            for (i in 0 until queries.rows) {
                assertEquals(10.0 + 3.0 * expected.means[i], actual.means[i], 1e-10)
                for (j in 0 until queries.rows) {
                    assertEquals(9.0 * expected.covariance[i, j], actual.covariance[i, j], 1e-10)
                }
            }
        }
    }

    @Test
    fun `log response conditioning scales known noise and retains unknown noise in training units`() {
        val fitted = GaussianProcessSurrogate(
            GaussianProcessConfig(mean = 0.0, learnNoise = true, hyperparameterTrials = 8),
        )
            .fit(data(rows(0.0, 0.8), doubleArrayOf(-1.0, 1.0), doubleArrayOf(0.0, Double.NaN)), Random(2))
        val wrapped = ResponseTransform(2.0, 3.0, logarithmic = true).model(fitted) as ConditionableSurrogate
        val responses = doubleArrayOf(exp(2.0), exp(3.5))
        val temporary = data(
            rows(0.2, 0.6),
            responses,
            doubleArrayOf(9.0 * responses[0] * responses[0] * 0.25, Double.NaN),
        )
        val actual = wrapped.condition(temporary) as LogResponseSurrogate
        val expected = fitted.condition(data(rows(0.2, 0.6), doubleArrayOf(0.0, 0.5), doubleArrayOf(0.25, Double.NaN)))
        val queries = rows(0.1, 0.5, 0.9)
        val expectedLog = expected.predictJoint(queries)
        val actualLog = assertNotNull(actual.predictLogJoint(queries))
        for (i in 0 until queries.rows) {
            assertEquals(2.0 + 3.0 * expectedLog.means[i], actualLog.means[i], 1e-10)
            for (j in 0 until queries.rows) {
                assertEquals(9.0 * expectedLog.covariance[i, j], actualLog.covariance[i, j], 1e-10)
            }
        }
        assertEquals(responses.toList(), temporary.responses.toDoubleArray().toList())
        assertTrue(temporary.noiseVariances!![1].isNaN())
    }

    @Test
    fun `mixed noise pending fantasies preserve real observations and reuse the fitted model`() {
        val projection = SearchTransform(Rate().compileSpace())
        val backend = WitnessSearch { request, _ ->
            val points = (1..8).map { continuousPoint(doubleArrayOf(it / 10.0)).sample }
                .filter { it !in request.exclude }
            WitnessBatch(points.take(request.count), WitnessTermination.BudgetExhausted)
        }
        val optimizer = BayesianOptimizer(
            projection,
            GaussianProcessSurrogate(GaussianProcessConfig(learnNoise = true, hyperparameterTrials = 8)),
            acquisitionOptimizer = LocalAndRandomSearch(projection, backend, candidateCount = 8, localSteps = 0),
            initialDesign = RandomInitialDesign(0, 0.0),
            batchPolicy = SequentialBatchPolicy(FantasyPolicy.PosteriorMean),
            outcome = OutcomeSpec(standardize = true, noise = ObservationNoise.Noisy),
        )
        optimizer.train(continuousPoint(doubleArrayOf(0.0)), Context.Empty, 2.0, noiseVariance = 0.25)
        optimizer.train(continuousPoint(doubleArrayOf(1.0)), 4.0)
        val pending = optimizer.suggest(1)[0]
        val fits = optimizer.fitCount
        val batch = optimizer.suggest(4)
        assertEquals(4, batch.size)
        assertEquals(fits, optimizer.fitCount)
        assertEquals(5, optimizer.pendingTrials.size)
        assertTrue(batch.none { it.sample == pending.sample })
        assertEquals(5, optimizer.pendingTrials.map { it.sample }.distinct().size)
        val history = optimizer.store.snapshot()
        assertEquals(2L, history.revision)
        assertEquals(listOf(2.0, 4.0), history.observations.map { it.reward })
        assertEquals(listOf(0.25, null), history.observations.map { it.noiseVariance })
    }

    @Test
    fun `conditioning rejects statistical weights even when mixed row noise is supplied`() {
        val fitted = GaussianProcessSurrogate(GaussianProcessConfig(learnNoise = true, hyperparameterTrials = 0))
            .fit(data(rows(0.0), doubleArrayOf(1.0), doubleArrayOf(Double.NaN)), Random(1))
        assertFailsWith<IllegalArgumentException> {
            fitted.condition(
                TrainingData(
                    rows(0.5),
                    DenseVector.of(doubleArrayOf(2.0)),
                    DenseVector.of(doubleArrayOf(2.0)),
                    DenseVector.of(doubleArrayOf(0.25)),
                ),
            )
        }
    }
}
