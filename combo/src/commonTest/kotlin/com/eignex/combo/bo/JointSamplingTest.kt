package com.eignex.combo.bo

import com.eignex.combo.bo.model.GaussianProcessConfig
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.kumulant.math.NotPositiveDefinite
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class JointSamplingTest {
    @Test
    fun `response decorators preserve covariance reference units for raw and log Gaussian sampling`() {
        val fitted = GaussianProcessSurrogate(
            GaussianProcessConfig(signalVariance = 2.0, hyperparameterTrials = 0),
        ).fit(
            TrainingData(jointPoints(0.0), DenseVector.of(doubleArrayOf(1.0)), DenseVector.of(doubleArrayOf(1.0))),
            Random(1),
        )
        for (scale in listOf(1e-3, 3.0, 1e3)) {
            val raw = ResponseTransform(
                2.0,
                scale,
            ).model(fitted).jointPosterior(jointPoints(0.1)) as JointPosteriorNumerics
            val log = (ResponseTransform(2.0, scale, logarithmic = true).model(fitted) as LogResponseSurrogate)
                .logJointPosterior(jointPoints(0.1)) as JointPosteriorNumerics
            assertEquals(2.0 * scale * scale, raw.covarianceScale)
            assertEquals(raw.covarianceScale, log.covarianceScale)
        }
    }

    @Test
    fun `conditional cancellation uses prior scale and still rejects material indefiniteness`() {
        fun posterior(scale: Double, cross: Double) = object : JointPosteriorNumerics {
            override val covarianceScale = scale
            override val baseline = JointPredictiveMoments(
                DenseVector.zero(1),
                DenseMatrix.wrap(1, 1, doubleArrayOf(1e-10)),
            )
            override fun predict(features: DenseMatrix) = JointPosteriorPrediction(
                JointPredictiveMoments(DenseVector.zero(1), DenseMatrix.wrap(1, 1, doubleArrayOf(1e-10))),
                DenseMatrix.wrap(1, 1, doubleArrayOf(cross)),
            )
        }
        // A nearly certain GP subtracts O(1) kernel terms to obtain O(1e-10) moments.
        // These rounded moments have a -2e-16 conditional residual, not negative uncertainty.
        val sampler = JointGaussianSampler(posterior(1.0, 1.000001e-10), 64, 7L)
        val draws = sampler.draw(jointPoints(0.3))
        assertTrue(draws.values.all { it.isFinite() })
        assertEquals(draws.values.toList(), sampler.draw(jointPoints(0.3)).values.toList())
        assertFailsWith<NotPositiveDefinite> {
            JointGaussianSampler(posterior(0.0, 1.000001e-10), 64, 7L).draw(jointPoints(0.3))
        }
        assertFailsWith<NotPositiveDefinite> {
            JointGaussianSampler(posterior(1.0, 1e-4), 64, 7L).draw(jointPoints(0.3))
        }
        assertFailsWith<IllegalArgumentException> { JointGaussianSampler(posterior(Double.NaN, 0.0), 64, 7L) }
    }

    @Test
    fun `tiny conditional factorization is stabilized relative to its kernel scale`() {
        val posterior = object : JointPosteriorNumerics {
            override val covarianceScale = 1.0
            override val baseline = JointPredictiveMoments(DenseVector.zero(0), DenseMatrix.zero(0))
            override fun predict(features: DenseMatrix) = JointPosteriorPrediction(
                JointPredictiveMoments(
                    DenseVector.zero(2),
                    DenseMatrix.wrap(2, 2, doubleArrayOf(1e-20, 1e-17, 1e-17, 1e-20)),
                ),
                DenseMatrix.zero(0, 2),
            )
        }
        val draws = JointGaussianSampler(posterior, 64, 11L).draw(jointPoints(0.2, 0.3))
        assertTrue(draws.values.all { it.isFinite() && kotlin.math.abs(it) < 1e-4 })
    }

    @Test
    fun `Sobol acquisition scores are fixed across batch size changes and candidate reorderings`() {
        val model = GaussianProcessSurrogate(GaussianProcessConfig(hyperparameterTrials = 0)).fit(
            TrainingData(
                jointPoints(0.0, 0.5, 1.0),
                DenseVector.of(doubleArrayOf(0.2, -0.1, 0.4)),
                DenseVector.of(doubleArrayOf(1.0, 1.0, 1.0)),
                DenseVector.of(doubleArrayOf(0.0009, 0.0009, 0.0009)),
            ),
            Random(1),
        )
        val context = JointAcquisitionContext(
            model,
            0.1,
            OutcomeSpec(),
            jointPoints(0.0, 0.5, 1.0),
            jointPoints(0.25),
        )
        val sobol = JointNormalSampling.ScrambledSobol
        val factories = listOf(
            JointExpectedImprovement(sobol, 128),
            JointNoisyExpectedImprovement(sobol, 128),
            JointLogExpectedImprovement(sobol, 128),
            JointLogNoisyExpectedImprovement(sobol, 128),
        )
        for (factory in factories) {
            val acquisition = factory.create(context, Random(13))
            val first = acquisition.evaluate(jointPoints(0.1, 0.3, 0.7, 0.9))
            assertTrue(first.isFinite())
            acquisition.evaluate(jointPoints(0.3))
            acquisition.evaluate(jointPoints(0.3, 0.9))
            assertEquals(first, acquisition.evaluate(jointPoints(0.9, 0.3, 0.1, 0.7, 0.3)))
            assertEquals(first, factory.create(context, Random(13)).evaluate(jointPoints(0.1, 0.3, 0.7, 0.9)))
        }
    }

    @Test
    fun `Sobol growth preserves baseline samples and leading conditional draws`() {
        val model = GaussianProcessSurrogate(GaussianProcessConfig(hyperparameterTrials = 0)).fit(
            TrainingData(jointPoints(0.0), DenseVector.of(doubleArrayOf(1.0)), DenseVector.of(doubleArrayOf(1.0))),
            Random(1),
        )
        val sampler = JointGaussianSampler(
            model.jointPosterior(jointPoints(0.1)),
            64,
            7L,
            JointNormalSampling.ScrambledSobol,
        )
        val baseline = sampler.baselineDraws.values.toList()
        val first = sampler.draw(jointPoints(0.3))
        val expanded = sampler.draw(jointPoints(0.3, 0.6, 0.7, 0.9))
        val repeated = sampler.draw(jointPoints(0.3))
        for (draw in 0 until 64) {
            assertEquals(first[0, draw], expanded[0, draw], 1e-12)
            assertEquals(first[0, draw], repeated[0, draw])
        }
        assertEquals(baseline, sampler.baselineDraws.values.toList())
        assertEquals(1, sampler.baselineFactorizations)
        assertEquals(3, sampler.candidateFactorizations)
    }

    @Test
    fun `Sobol requires powers of two without tightening antithetic sample counts`() {
        val sobol = JointNormalSampling.ScrambledSobol
        val constructors: List<(Int) -> JointAcquisitionFactory> = listOf(
            { JointExpectedImprovement(sobol, it) },
            { JointNoisyExpectedImprovement(sobol, it) },
            { JointLogExpectedImprovement(sobol, it) },
            { JointLogNoisyExpectedImprovement(sobol, it) },
        )
        for (create in constructors) {
            assertFailsWith<IllegalArgumentException> { create(6) }
            create(8)
        }
        assertEquals(JointNormalSampling.Antithetic, JointExpectedImprovement(6).sampling)
        assertEquals(JointNormalSampling.Antithetic, JointNoisyExpectedImprovement(6).sampling)
        assertEquals(JointNormalSampling.Antithetic, JointLogExpectedImprovement(6).sampling)
        assertEquals(JointNormalSampling.Antithetic, JointLogNoisyExpectedImprovement(6).sampling)
    }
}
