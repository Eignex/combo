package com.eignex.combo.bo

import com.eignex.combo.test.TimedTest
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import kotlin.math.PI
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ExpectedImprovementTest : TimedTest() {
    private fun score(
        mean: Double,
        variance: Double,
        incumbent: Double,
        maximize: Boolean = true,
        margin: Double = 0.0,
        logarithmic: Boolean = false,
    ): Double {
        val model = object : FittedSurrogate {
            override val uncertaintyKind = UncertaintyKind.LatentPosterior
            override fun predict(features: DenseMatrix) = PredictiveMoments(
                DenseVector.of(DoubleArray(features.rows) { mean }),
                DenseVector.of(DoubleArray(features.rows) { variance }),
            )
            override fun drawFunction(random: Random): SampledFunction = error("EI does not draw functions")
        }
        val fitted = if (logarithmic) ResponseTransform(0.0, 1.0, logarithmic = true).model(model) else model
        return ExpectedImprovement(margin).create(
            AcquisitionContext(fitted, incumbent, OutcomeSpec(maximize = maximize)),
            Random(1),
        ).evaluate(DenseMatrix.zero(1, 1))[0]
    }

    @Test
    fun `lognormal cost improvement remains bounded when raw moments overflow`() {
        val value = score(0.0, 1e6, 1.0, maximize = false, logarithmic = true)
        assertTrue(value.isFinite() && value in 0.49..0.5)
    }

    @Test
    fun `lognormal expected improvement agrees with integrated log response moments`() {
        assertEquals(0.2384217081, score(0.0, 1.0, 1.0, maximize = false, logarithmic = true), 1e-6)
        assertEquals(0.8871429788, score(0.0, 1.0, 1.0, logarithmic = true), 1e-6)
    }

    @Test
    fun `lognormal deterministic improvement uses original units and margin`() {
        assertEquals(1.0, score(ln(2.0), 0.0, 3.0, maximize = false, logarithmic = true))
        assertEquals(0.5, score(ln(2.0), 0.0, 3.0, maximize = false, margin = 0.5, logarithmic = true))
        assertEquals(1.0, score(ln(2.0), 0.0, 1.0, logarithmic = true))
        assertEquals(0.0, score(ln(2.0), 0.0, 1.0, maximize = false, margin = 2.0, logarithmic = true))
    }

    @Test
    fun `zero variance expected improvement is deterministic improvement`() {
        assertEquals(1.0, score(3.0, 0.0, 2.0))
        assertEquals(0.0, score(1.0, 0.0, 2.0))
        assertEquals(1.0, score(1.0, 0.0, 2.0, maximize = false))
    }

    @Test
    fun `uncertainty gives positive improvement at the incumbent mean`() {
        assertEquals(2.0 / sqrt(2.0 * PI), score(2.0, 4.0, 2.0), 1e-12)
    }

    @Test
    fun `Gaussian expected improvement agrees with a known value`() {
        assertEquals(0.0833154706, score(0.0, 1.0, 1.0), 1e-7)
    }

    @Test
    fun `minimization is equivalent to sign flipped maximization`() {
        val minimizing = score(1.0, 0.5, 2.0, maximize = false)
        val maximizing = score(-1.0, 0.5, -2.0)
        assertEquals(minimizing, maximizing, 1e-12)
    }

    @Test
    fun `improvement margin reduces acquisition value`() {
        assertTrue(score(1.0, 1.0, 1.0, margin = 0.5) < score(1.0, 1.0, 1.0))
    }
}
