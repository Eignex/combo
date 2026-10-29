package com.eignex.combo.bo

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.kumulant.math.NotPositiveDefinite
import kotlin.math.PI
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal fun jointPoints(vararg x: Double): DenseMatrix = DenseMatrix.wrap(x.size, 1, x)
private class GaussianToy(
    private val mean: (Double) -> Double = { 0.0 },
    private val covariance: (Double, Double) -> Double = { a, b -> if (a == b) 1.0 else 0.0 },
) : FittedSurrogate {
    override val uncertaintyKind = UncertaintyKind.LatentPosterior
    override fun predictJoint(features: DenseMatrix): JointPredictiveMoments = JointPredictiveMoments(
        DenseVector.of(DoubleArray(features.rows) { mean(features[it, 0]) }),
        DenseMatrix.zero(features.rows).also { matrix ->
            for (i in 0 until features.rows) {
                for (j in 0 until features.rows) {
                    matrix[i, j] = covariance(features[i, 0], features[j, 0])
                }
            }
        },
    )
    override fun predict(features: DenseMatrix): PredictiveMoments {
        val joint = predictJoint(features)
        return PredictiveMoments(joint.means, DenseVector.of(DoubleArray(features.rows) { joint.covariance[it, it] }))
    }
    override fun drawFunction(random: Random): SampledFunction = error("joint EI uses fixed Gaussian base samples")
}

class JointAcquisitionTest {
    private fun context(
        model: FittedSurrogate = GaussianToy(),
        maximize: Boolean = true,
        pending: DenseMatrix = jointPoints(),
    ) = JointAcquisitionContext(model, 0.0, OutcomeSpec(maximize = maximize), jointPoints(-1.0), pending)

    @Test
    fun `one point joint EI agrees with analytic EI for both orientations`() {
        for (maximize in listOf(false, true)) {
            val acquisition = JointExpectedImprovement(samples = 256).create(context(maximize = maximize), Random(3))
            assertEquals(1.0 / sqrt(2.0 * PI), acquisition.evaluate(jointPoints(1.0)), 0.12)
        }
    }

    @Test
    fun `joint utility uses cross point correlation instead of adding marginal EIs`() {
        val factory = JointExpectedImprovement(samples = 256)
        val positive = GaussianToy(covariance = { _, _ -> 1.0 })
        val negative = GaussianToy(covariance = { a, b -> if (a == b) 1.0 else -1.0 })
        val together = factory.create(context(positive), Random(7)).evaluate(jointPoints(1.0, 2.0))
        val opposite = factory.create(context(negative), Random(7)).evaluate(jointPoints(1.0, 2.0))

        assertEquals(1.0 / sqrt(2.0 * PI), together, 0.12)
        assertEquals(sqrt(2.0 / PI), opposite, 0.16)
        assertTrue(opposite > together * 1.5)
    }

    @Test
    fun `noisy EI integrates the uncertain latent incumbent`() {
        val acquisition = JointNoisyExpectedImprovement(samples = 256).create(context(), Random(13))

        assertEquals(1.0 / sqrt(PI), acquisition.evaluate(jointPoints(1.0)), 0.12)
        assertEquals(0.0, acquisition.evaluate(jointPoints(-1.0)))
    }

    @Test
    fun `candidate order duplicates and repeated evaluation preserve the same bound objective`() {
        val acquisition = JointExpectedImprovement(samples = 512).create(context(), Random(19))
        val first = acquisition.evaluate(jointPoints(1.0, 2.0))

        assertEquals(first, acquisition.evaluate(jointPoints(2.0, 1.0)))
        assertEquals(first, acquisition.evaluate(jointPoints(1.0, 2.0, 1.0)))
        acquisition.evaluate(jointPoints(3.0))
        assertEquals(first, acquisition.evaluate(jointPoints(1.0, 2.0)))
        assertEquals(0.0, acquisition.evaluate(jointPoints()))
    }

    @Test
    fun `pending improvement is excluded from incremental batch utility and inputs are captured`() {
        val pending = jointPoints(1.0)
        val acquisition = JointExpectedImprovement(samples = 512).create(context(pending = pending), Random(23))
        pending.values[0] = 99.0

        assertEquals(0.0, acquisition.evaluate(jointPoints(1.0)))
        assertTrue(acquisition.evaluate(jointPoints(2.0)) > 0.0)
    }

    @Test
    fun `zero covariance stays deterministic without artificial improvement from jitter`() {
        val model = GaussianToy(mean = { it }, covariance = { _, _ -> 0.0 })
        val acquisition = JointExpectedImprovement(margin = 0.5).create(context(model), Random(1))

        assertEquals(1.5, acquisition.evaluate(jointPoints(1.0, 2.0)))
        assertEquals(0.0, acquisition.evaluate(jointPoints(-1.0)))
    }

    @Test
    fun `log response joint moments are sampled in log coordinates`() {
        val normal = GaussianToy(mean = { ln(2.0) }, covariance = { _, _ -> 0.0 })
        val model = object : LogResponseSurrogate, FittedSurrogate by normal {
            override fun predictLog(features: DenseMatrix) = normal.predict(features)
            override fun predictLogJoint(features: DenseMatrix) = normal.predictJoint(features)
        }
        val acquisition = JointExpectedImprovement().create(
            JointAcquisitionContext(model, 3.0, OutcomeSpec(maximize = false), jointPoints(-1.0)),
            Random(1),
        )

        assertEquals(1.0, acquisition.evaluate(jointPoints(1.0)))
    }

    @Test
    fun `invalid settings missing joint moments and materially indefinite covariance are rejected`() {
        assertFailsWith<IllegalArgumentException> { JointExpectedImprovement(samples = 3) }
        assertFailsWith<IllegalArgumentException> { JointNoisyExpectedImprovement(margin = -1.0) }
        val marginal = object : FittedSurrogate by GaussianToy() {
            override fun predictJoint(features: DenseMatrix): JointPredictiveMoments? = null
        }
        assertFailsWith<IllegalArgumentException> { JointExpectedImprovement().create(context(marginal), Random(1)) }
        val invalid = GaussianToy(covariance = { a, b -> if (a == b) 1.0 else 2.0 })
        val acquisition = JointExpectedImprovement().create(context(invalid), Random(1))
        assertFailsWith<NotPositiveDefinite> { acquisition.evaluate(jointPoints(1.0, 2.0)) }
    }
}
