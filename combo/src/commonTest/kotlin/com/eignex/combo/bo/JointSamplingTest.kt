package com.eignex.combo.bo

import com.eignex.combo.bo.model.GaussianProcessConfig
import com.eignex.combo.test.TimedTest
import com.eignex.koblas.DenseVector
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class JointSamplingTest : TimedTest() {
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
