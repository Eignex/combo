package com.eignex.combo.bo

import com.eignex.combo.bo.model.GaussianHyperparameterSearch
import com.eignex.combo.bo.model.GaussianMeanEstimation
import com.eignex.combo.bo.model.GaussianProcessConfig
import com.eignex.combo.bo.model.GaussianProcessFitting
import com.eignex.combo.bo.model.GaussianProcessRegularization
import com.eignex.combo.test.TimedTest
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class GaussianProcessFittingTest : TimedTest() {
    private fun training() = TrainingData(
        DenseMatrix.ofRows(arrayOf(doubleArrayOf(0.0), doubleArrayOf(100.0))),
        DenseVector.of(doubleArrayOf(2.0, 7.0)),
        DenseVector.of(doubleArrayOf(1.0, 1.0)),
        DenseVector.of(doubleArrayOf(0.0, 3.0)),
    )
    private val profiled = GaussianProcessFitting(meanEstimation = GaussianMeanEstimation.CovarianceWeighted)
    private val fixed = GaussianProcessConfig(hyperparameterTrials = 0, jitter = 1e-12)

    @Test
    fun `profiled constant mean agrees with independent inverse covariance reference`() {
        val empirical = GaussianProcessSurrogate(fixed).fit(training(), Random(1))
        val fit = GaussianProcessSurrogate(profiled, fixed).fit(training(), Random(1))
        assertEquals(4.5, empirical.parameters.mean)
        assertEquals(3.0, fit.parameters.mean, 1e-10)
        assertTrue(fit.logMarginalLikelihood > empirical.logMarginalLikelihood)
        assertEquals(fit.logMarginalLikelihood, fit.logFitScore)
    }

    @Test
    fun `explicit mean overrides profiling and conditioning retains estimated parameters`() {
        val explicit = GaussianProcessSurrogate(profiled, fixed.copy(mean = 12.0)).fit(training(), Random(1))
        assertEquals(12.0, explicit.parameters.mean)
        val fit = GaussianProcessSurrogate(profiled, fixed).fit(training(), Random(1))
        val temporary = TrainingData(
            DenseMatrix.ofRows(arrayOf(doubleArrayOf(50.0))),
            DenseVector.of(doubleArrayOf(100.0)),
            DenseVector.of(doubleArrayOf(1.0)),
            DenseVector.of(doubleArrayOf(0.0)),
        )
        assertEquals(fit.parameters, fit.condition(temporary).parameters)
        val regularized = GaussianProcessSurrogate(
            profiled.copy(regularization = GaussianProcessRegularization()),
            fixed,
        ).fit(training(), Random(1))
        val conditioned = regularized.condition(temporary)
        assertEquals(
            regularized.logFitScore - regularized.logMarginalLikelihood,
            conditioned.logFitScore - conditioned.logMarginalLikelihood,
            1e-10,
        )
    }

    @Test
    fun `known row noise is never penalized or fitted as unknown noise`() {
        val prior = GaussianProcessRegularization(noiseVariance = 1e-9)
        val left = GaussianProcessSurrogate(
            GaussianProcessFitting(regularization = prior),
            fixed.copy(learnNoise = true),
        ).fit(training(), Random(1))
        val right = GaussianProcessSurrogate(
            GaussianProcessFitting(regularization = prior.copy(noiseVariance = 1e9)),
            fixed.copy(learnNoise = true),
        ).fit(training(), Random(1))
        assertEquals(left.logFitScore, right.logFitScore)
        assertEquals(left.parameters, right.parameters)
    }

    @Test
    fun `seeded multistart retains initial score and is reproducible`() {
        val policy = profiled.copy(
            regularization = GaussianProcessRegularization(),
            search = GaussianHyperparameterSearch.Multistart,
        )
        val initial = GaussianProcessSurrogate(policy, fixed).fit(training(), Random(71))
        val surrogate = GaussianProcessSurrogate(policy, fixed.copy(hyperparameterTrials = 37))
        val first = surrogate.fit(training(), Random(71))
        val second = surrogate.fit(training(), Random(71))
        assertTrue(first.logFitScore >= initial.logFitScore)
        assertEquals(first.parameters, second.parameters)
        assertEquals(first.logFitScore, second.logFitScore)
        assertEquals(policy, surrogate.fitting)
    }

    @Test
    fun `unknown noise regularization affects only its fitted covariance parameter`() {
        val data = training().let { TrainingData(it.features, it.responses, it.weights) }
        val config = fixed.copy(noiseVariance = 1.0, learnNoise = true, hyperparameterTrials = 64)
        fun fit(center: Double) = GaussianProcessSurrogate(
            GaussianProcessFitting(
                regularization = GaussianProcessRegularization(
                    noiseVariance = center,
                    noiseLogDeviation = 0.1,
                ),
            ),
            config,
        ).fit(data, Random(1))
        val quiet = fit(0.25)
        val noisy = fit(4.0)
        assertTrue(noisy.parameters.noiseVariance > quiet.parameters.noiseVariance)
        assertTrue(noisy.logFitScore <= noisy.logMarginalLikelihood)
    }

    @Test
    fun `zero candidate budget leaves supplied covariance unchanged for every search policy`() {
        for (search in GaussianHyperparameterSearch.entries) {
            val fit = GaussianProcessSurrogate(profiled.copy(search = search), fixed).fit(training(), Random(1))
            assertEquals(listOf(fixed.lengthScale), fit.parameters.lengthScales)
            assertEquals(fixed.signalVariance, fit.parameters.signalVariance)
        }
    }

    @Test
    fun `invalid fitting policies fail before training`() {
        assertFailsWith<IllegalArgumentException> { GaussianProcessFitting(starts = 0) }
        assertFailsWith<IllegalArgumentException> { GaussianProcessRegularization(lengthScale = 0.0) }
        assertFailsWith<IllegalArgumentException> { GaussianProcessRegularization(noiseLogDeviation = Double.NaN) }
    }
}
