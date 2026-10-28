package com.eignex.combo.bo

import com.eignex.combo.bo.model.GaussianKernel
import com.eignex.combo.bo.model.GaussianProcessConfig
import com.eignex.combo.test.TimedTest
import com.eignex.klause.schema.le
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import kotlin.math.exp
import kotlin.math.sin
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class GaussianProcessSurrogateTest : TimedTest() {
    private val fixed = GaussianProcessConfig(
        kernel = GaussianKernel.Rbf,
        lengthScale = 1.0,
        mean = 0.0,
        noiseVariance = 0.0,
        hyperparameterTrials = 0,
        jitter = 1e-12,
    )
    private fun rows(vararg x: Double) = DenseMatrix.ofRows(x.map { doubleArrayOf(it) }.toTypedArray())
    private fun data(x: DenseMatrix, y: DoubleArray, noise: DoubleArray? = null) = TrainingData(
        x,
        DenseVector.of(y),
        DenseVector.of(DoubleArray(y.size) { 1.0 }),
        noise?.let(DenseVector::of),
    )
    private fun fit(data: TrainingData, config: GaussianProcessConfig = fixed) = GaussianProcessSurrogate(
        config,
    ).fit(data, Random(1))

    @Test
    fun `one observation agrees with analytic Gaussian conditioning including joint covariance`() {
        val fitted = fit(data(rows(0.0), doubleArrayOf(2.0), doubleArrayOf(1.0)))
        val marginal = fitted.predict(rows(0.0, 1.0))
        val joint = fitted.predictJoint(rows(0.0, 1.0))
        assertEquals(1.0, marginal.means[0], 1e-10)
        assertEquals(exp(-0.5), marginal.means[1], 1e-10)
        assertEquals(0.5, marginal.variances[0], 1e-10)
        assertEquals(1.0 - 0.5 * exp(-1.0), marginal.variances[1], 1e-10)
        assertEquals(0.5 * exp(-0.5), joint.covariance[0, 1], 1e-10)
        for (i in 0..1) {
            assertEquals(marginal.means[i], joint.means[i])
            assertEquals(marginal.variances[i], joint.covariance[i, i], 1e-12)
        }
    }

    @Test
    fun `empty training data predicts the configured prior`() {
        val fitted = fit(data(DenseMatrix.zero(0, 1), doubleArrayOf()))
        val joint = fitted.predictJoint(rows(0.0, 1.0))
        assertEquals(listOf(0.0, 0.0), joint.means.toDoubleArray().toList())
        assertEquals(1.0, joint.covariance[0, 0])
        assertEquals(exp(-0.5), joint.covariance[0, 1], 1e-12)
        assertEquals(0.0, fitted.logMarginalLikelihood)
        assertEquals(0, fitted.predict(DenseMatrix.zero(0, 1)).means.size)
    }

    @Test
    fun `known observation noise is separate from latent uncertainty and numerical jitter`() {
        val quiet = fit(data(rows(0.0), doubleArrayOf(2.0), doubleArrayOf(0.01)))
        val noisy = fit(data(rows(0.0), doubleArrayOf(2.0), doubleArrayOf(4.0)))
        assertTrue(quiet.predict(rows(0.0)).means[0] > noisy.predict(rows(0.0)).means[0])
        assertTrue(quiet.predict(rows(0.0)).variances[0] < noisy.predict(rows(0.0)).variances[0])
        assertEquals(0.0, noisy.parameters.noiseVariance)
        val known = fit(data(rows(0.0), doubleArrayOf(2.0), doubleArrayOf(4.0)), fixed.copy(learnNoise = true))
        assertEquals(0.0, known.parameters.noiseVariance)
        assertEquals(noisy.predict(rows(0.0)).means[0], known.predict(rows(0.0)).means[0])
        assertTrue(noisy.numericalJitter < 1e-8)
    }

    @Test
    fun `statistical weights are never interpreted as replicate counts or noise`() {
        assertFailsWith<IllegalArgumentException> {
            fit(TrainingData(rows(0.0), DenseVector.of(doubleArrayOf(1.0)), DenseVector.of(doubleArrayOf(2.0))))
        }
    }

    @Test
    fun `conditioning matches a fixed parameter combined fit and leaves the original model unchanged`() {
        val first = data(rows(0.0), doubleArrayOf(1.0), doubleArrayOf(0.1))
        val fitted = fit(first)
        val before = fitted.predict(rows(0.3)).means[0]
        val conditioned = fitted.condition(data(rows(1.0), doubleArrayOf(3.0), doubleArrayOf(0.2)))
        val combined = fit(data(rows(0.0, 1.0), doubleArrayOf(1.0, 3.0), doubleArrayOf(0.1, 0.2)))
        val actual = conditioned.predictJoint(rows(0.3, 0.7))
        val expected = combined.predictJoint(rows(0.3, 0.7))
        assertEquals(expected.means.toDoubleArray().toList(), actual.means.toDoubleArray().toList())
        assertEquals(expected.covariance.values.toList(), actual.covariance.values.toList())
        assertEquals(before, fitted.predict(rows(0.3)).means[0])
        assertEquals(fitted.parameters, conditioned.parameters)
    }

    @Test
    fun `likelihood search improves or retains its initial fit reproducibly`() {
        val x = DoubleArray(12) { it / 11.0 }
        val training = data(rows(*x), DoubleArray(x.size) { sin(5.0 * x[it]) })
        val initial = GaussianProcessConfig(lengthScale = 0.03, hyperparameterTrials = 0, learnNoise = true)
        val baseline = fit(training, initial)
        val optimized = fit(training, initial.copy(hyperparameterTrials = 32))
        assertTrue(optimized.logMarginalLikelihood > baseline.logMarginalLikelihood)
        assertEquals(optimized.parameters, fit(training, initial.copy(hyperparameterTrials = 32)).parameters)
        assertTrue(optimized.parameters.noiseVariance > 0.0)
    }

    @Test
    fun `fitting captures mutable training and parameter storage`() {
        val training = data(rows(0.0, 1.0), doubleArrayOf(1.0, 3.0))
        val scales = mutableListOf(1.0)
        val fitted = fit(training, fixed.copy(lengthScales = scales))
        val before = fitted.predict(rows(0.3)).means[0]
        training.features[0, 0] = 10.0
        (training.responses as DenseVector)[0] = 10.0
        scales[0] = 10.0
        assertEquals(before, fitted.predict(rows(0.3)).means[0])
    }

    @Test
    fun `duplicate observations and distant finite inputs remain numerically finite`() {
        for (kernel in GaussianKernel.entries) {
            val fitted = fit(data(rows(0.0, 0.0), doubleArrayOf(1.0, 1.0)), fixed.copy(kernel = kernel))
            val prediction = fitted.predict(rows(0.0, 1e308))
            assertTrue(prediction.means.toDoubleArray().all { it.isFinite() })
            assertTrue(prediction.variances.toDoubleArray().all { it.isFinite() && it >= 0.0 })
            assertEquals(fixed.signalVariance, prediction.variances[1])
        }
    }

    @Test
    fun `sampled functions retain cached values across duplicate reordered and incremental queries`() {
        val fitted = fit(data(rows(0.0), doubleArrayOf(1.0), doubleArrayOf(0.1)))
        val draw = fitted.drawFunction(Random(7))
        val first = draw.evaluate(rows(0.2, 0.8, 0.2))
        assertEquals(first[0], first[2])
        val second = draw.evaluate(rows(0.8, 0.5, 0.2))
        assertEquals(first[0], second[2])
        assertEquals(first[1], second[0])
        assertEquals(second[1], draw.evaluate(rows(0.5))[0])
        assertEquals(
            first.toDoubleArray().toList(),
            fitted.drawFunction(Random(7)).evaluate(rows(0.2, 0.8, 0.2)).toDoubleArray().toList(),
        )
    }

    @Test
    fun `joint sampled values preserve posterior cross point dependence`() {
        val fitted = fit(data(DenseMatrix.zero(0, 1), doubleArrayOf()))
        var left = 0.0
        var right = 0.0
        var product = 0.0
        val queries = rows(0.0, 0.4)
        repeat(1500) {
            val values = fitted.drawFunction(Random(it)).evaluate(queries)
            left += values[0]
            right += values[1]
            product += values[0] * values[1]
        }
        val covariance = product / 1500 - left * right / (1500.0 * 1500.0)
        assertEquals(exp(-0.08), covariance, 0.09)
    }

    @Test
    fun `response transforms preserve joint covariance through conditioning and log moments`() {
        val fitted = fit(data(rows(0.0), doubleArrayOf(1.0), doubleArrayOf(0.1)))
        val queries = rows(0.2, 0.8)
        val original = fitted.predictJoint(queries)
        val affine = ResponseTransform(2.0, 3.0).model(fitted)
        val joint = assertNotNull(affine.predictJoint(queries))
        assertEquals(2.0 + 3.0 * original.means[0], joint.means[0], 1e-12)
        assertEquals(9.0 * original.covariance[0, 1], joint.covariance[0, 1], 1e-12)
        val logged = ResponseTransform(2.0, 3.0, logarithmic = true).model(fitted) as LogResponseSurrogate
        val logJoint = assertNotNull(logged.predictLogJoint(queries))
        assertEquals(joint.covariance[0, 1], logJoint.covariance[0, 1])
        val raw = assertNotNull(logged.predictJoint(queries))
        val marginal = logged.predict(queries)
        assertEquals(marginal.variances[0], raw.covariance[0, 0], 1e-10)
        val conditioned = (logged as ConditionableSurrogate).condition(data(rows(1.0), doubleArrayOf(exp(2.0))))
        assertNotNull((conditioned as LogResponseSurrogate).predictLogJoint(queries))
    }

    @Test
    fun `invalid query dimensions or nonfinite coordinates fail before prediction`() {
        val fitted = fit(data(rows(0.0), doubleArrayOf(1.0)))
        assertFailsWith<IllegalArgumentException> { fitted.predict(DenseMatrix.zero(1, 2)) }
        assertFailsWith<IllegalArgumentException> { fitted.predictJoint(rows(Double.NaN)) }
        assertFailsWith<IllegalArgumentException> {
            fitted.condition(
                data(rows(Double.POSITIVE_INFINITY), doubleArrayOf(1.0)),
            )
        }
    }
    private class Rate : com.eignex.combo.decisions.DecisionSpace() {
        val x by floatVar(0.0, 1.0)
        val cap by constraint { x le 0.8 }
    }

    @Test
    fun `GP EI uses frozen pending fantasies without recording invented observations`() {
        val space = Rate().compileSpace()
        val optimizer = BayesianOptimizer(
            SearchTransform(space),
            GaussianProcessSurrogate(fixed),
            acquisitionOptimizer = LocalAndRandomSearch(
                SearchTransform(space),
                com.eignex.combo.shared.WitnessSearch { request, _ ->
                    val points = (1..6).map { continuousPoint(doubleArrayOf(it / 10.0)).sample }
                        .filter { it !in request.exclude }
                    com.eignex.combo.shared.WitnessBatch(
                        points.take(request.count),
                        com.eignex.combo.shared.WitnessTermination.CountReached,
                    )
                },
                candidateCount = 6,
                localSteps = 0,
            ),
            outcome = OutcomeSpec(standardize = true, noise = ObservationNoise.Noisy),
            initialDesign = RandomInitialDesign(observations = 1, probability = 0.0),
            batchPolicy = SequentialBatchPolicy(FantasyPolicy.PosteriorMean),
        )
        val first = optimizer.suggest(1)[0]
        optimizer.complete(first.id, 2.0, noiseVariance = 0.1)
        val pending = optimizer.suggest(1)[0]
        val fits = optimizer.fitCount
        val batch = optimizer.suggest(3)
        assertEquals(3, batch.size)
        assertEquals(fits, optimizer.fitCount)
        assertEquals(1, optimizer.store.snapshot().observations.size)
        assertEquals(4, optimizer.trials.count { it.status == TrialStatus.Pending })
        assertTrue(batch.none { it.sample == pending.sample })
        for (trial in batch) assertTrue(com.eignex.combo.shared.WitnessValidator(space).valid(trial.sample.sample))
    }

    @Test
    fun `conditioning freezes learned mean and covariance parameters`() {
        val fitted = fit(
            data(rows(0.0, 0.5, 1.0), doubleArrayOf(1.0, 3.0, 2.0)),
            GaussianProcessConfig(learnNoise = true),
        )
        val conditioned = fitted.condition(data(rows(0.3), doubleArrayOf(100.0), doubleArrayOf(10.0)))
        assertEquals(fitted.parameters, conditioned.parameters)
    }
}
