package com.eignex.combo.bench.diagnostics

import com.eignex.combo.bo.*
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import kotlin.math.*
import kotlin.random.Random
import kotlin.test.*

class ResearchPosteriorTest {
    private fun rows(vararg x: Double) = matrix(x.map { doubleArrayOf(it) }, 1)
    private fun data(x: DoubleArray, y: DoubleArray, noise: DoubleArray) = training(rows(*x), y, noise)
    @Test
    fun `Gaussian constant mean integration adds shared covariance and matches analytic conditioning`() {
        val prior = IntegratedMeanGp(data(doubleArrayOf(), doubleArrayOf(), doubleArrayOf()), listOf(1.0), 1.0, 0.0)
        val covariance = prior.predictJoint(rows(0.0, 100.0)).covariance
        assertEquals(2.0, covariance[0, 0])
        assertEquals(1.0, covariance[0, 1], 1e-12)
        val posterior = prior.condition(data(doubleArrayOf(0.0), doubleArrayOf(2.0), doubleArrayOf(1.0)))
        val p = posterior.predictJoint(rows(0.0, 100.0))
        assertEquals(4.0 / 3.0, p.means[0], 1e-8)
        assertEquals(2.0 / 3.0, p.means[1], 1e-8)
        assertEquals(2.0 / 3.0, p.covariance[0, 0], 1e-8)
        assertEquals(1.0 / 3.0, p.covariance[0, 1], 1e-8)
        assertEquals(5.0 / 3.0, p.covariance[1, 1], 1e-8)
    }
    @Test
    fun `zero mean prior agrees with the production fixed parameter GP`() {
        val data = data(doubleArrayOf(0.0, 0.4, 1.0), doubleArrayOf(1.0, -1.0, 2.0), doubleArrayOf(0.0, Double.NaN, 0.1))
        val reference = GaussianProcessSurrogate(com.eignex.combo.bo.model.GaussianProcessConfig(
            lengthScale = 0.5, signalVariance = 2.0, noiseVariance = 0.2, mean = 0.0, hyperparameterTrials = 0,
        )).fit(data, Random(1))
        val actual = IntegratedMeanGp(data, listOf(0.5), 2.0, 0.2, meanVariance = 0.0)
        val expected = reference.predictJoint(rows(0.1, 0.6))
        val prediction = actual.predictJoint(rows(0.1, 0.6))
        assertEquals(reference.logMarginalLikelihood, actual.logLikelihood, 1e-12)
        for (i in 0..1) {
            assertEquals(expected.means[i], prediction.means[i], 1e-12)
            for (j in 0..1) assertEquals(expected.covariance[i,j], prediction.covariance[i,j], 1e-12)
        }
    }
    @Test
    fun `affine joint caches preserve fixed baseline and cross covariance`() {
        val data = data(doubleArrayOf(0.0, 0.7), doubleArrayOf(-1.0, 1.0), doubleArrayOf(0.1, 0.2))
        val model = AffinePosterior(IntegratedMeanGp(data, listOf(0.5), 2.0, 0.1), 2.0, 3.0)
        val baseline = rows(0.1, 0.3)
        val cache = assertNotNull(model.jointPosterior(baseline))
        baseline[0,0] = 100.0
        val expected = assertNotNull(model.predictJoint(rows(0.1, 0.3, 0.8, 0.9)))
        val actual = cache.predict(rows(0.8, 0.9))
        for (i in 0..1) {
            assertEquals(expected.means[i], cache.baseline.means[i], 1e-12)
            assertEquals(expected.means[i+2], actual.candidates.means[i], 1e-12)
            for (j in 0..1) {
                assertEquals(expected.covariance[i,j+2], actual.crossCovariance[i,j], 1e-12)
                assertEquals(expected.covariance[i+2,j+2], actual.candidates.covariance[i,j], 1e-12)
            }
        }
    }
    @Test
    fun `mixture conditioning updates posterior weights by predictive evidence without changing the original`() {
        val training = data(doubleArrayOf(0.0), doubleArrayOf(1.0), doubleArrayOf(0.1))
        val a = IntegratedMeanGp(training, listOf(0.1), 0.5, 0.1)
        val b = IntegratedMeanGp(training, listOf(1.0), 2.0, 0.1)
        val mixture = ParameterMixture(listOf(a, b), doubleArrayOf(0.25, 0.75))
        val conditioned = mixture.condition(data(doubleArrayOf(0.5), doubleArrayOf(3.0), doubleArrayOf(0.2)))
        val expected = normalizedLogWeights(doubleArrayOf(ln(0.25) + conditioned.components[0].logLikelihood - a.logLikelihood,
            ln(0.75) + conditioned.components[1].logLikelihood - b.logLikelihood))
        assertContentEquals(expected, conditioned.weights)
        assertContentEquals(doubleArrayOf(0.25, 0.75), mixture.weights)
        assertNotEquals(mixture.weights[0], conditioned.weights[0])
        val draw = conditioned.drawFunction(Random(7))
        val first = draw.evaluate(rows(0.2, 0.8, 0.2))
        val second = draw.evaluate(rows(0.8, 0.5, 0.2))
        assertEquals(first[0], first[2])
        assertEquals(first[0], second[2])
        assertEquals(first[1], second[0])
    }
    @Test
    fun `grid posterior captures data and rejects non-unit weights during conditioning`() {
        val training = data(doubleArrayOf(0.0, 1.0), doubleArrayOf(-1.0, 1.0), doubleArrayOf(0.1, Double.NaN))
        val mixture = parameterGrid(training, 3)
        assertEquals(27, mixture.components.size)
        assertEquals(1.0, mixture.weights.sum(), 1e-12)
        val before = mixture.predict(rows(0.5)).means[0]
        training.features[0, 0] = 100.0
        (training.responses as DenseVector)[0] = 100.0
        assertEquals(before, mixture.predict(rows(0.5)).means[0])
        assertFailsWith<IllegalArgumentException> {
            mixture.condition(TrainingData(rows(0.5), DenseVector.of(doubleArrayOf(2.0)), DenseVector.of(doubleArrayOf(2.0))))
        }
    }
    @Test
    fun `joint acquisition integrates component utility with one common frozen random stream`() {
        val fitter = DiagnosticSurrogate("mixture", 0, true, 3)
        val model = fitter.fit(data(doubleArrayOf(0.0, 1.0), doubleArrayOf(0.0, 1.0), doubleArrayOf(Double.NaN, 0.1)), Random(8))
        val context = JointAcquisitionContext(model, 0.0, OutcomeSpec(maximize = false, noise = ObservationNoise.Noisy), rows(0.0, 1.0), rows(0.3))
        val actual = fitter.jointFactory(true, 8).create(context, Random(9))
        val seed = Random(9).nextLong()
        val reference = fitter.components().map { (component, weight) -> weight to JointLogNoisyExpectedImprovement(8).create(
            JointAcquisitionContext(component, context.incumbent, context.outcome, context.baseline, context.pending), Random(seed)) }
        val candidates = rows(0.5, 0.7)
        val values = reference.map { ln(it.first) + it.second.evaluate(candidates) }
        val max = values.max()
        assertEquals(max + ln(values.sumOf { exp(it - max) }), actual.evaluate(candidates), 1e-12)
        assertEquals(actual.evaluate(candidates), actual.evaluate(candidates))
    }
}
