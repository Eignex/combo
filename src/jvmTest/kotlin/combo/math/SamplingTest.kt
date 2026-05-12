package combo.math

import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse

class SamplingTest {
    @Test
    fun fixedSeed() {
        val r1 = Random(0)
        val r2 = Random(0)
        for (i in 0..100) {
            assertEquals(r1.nextNormal(), r2.nextNormal())
        }
    }

    @Test
    fun normalStats() {
        val r = Random(120)
        val s = generateSequence { r.nextNormal(mean = 1.0f, std = sqrt(2.0f)) }
                .take(200)
                .sample(RunningVariance())
        assertEquals(2.0f, s.variance, 0.5f)
        assertEquals(1.0f, s.mean, 0.2f)
    }

    @Test
    fun gammaShapeLessThan1() {
        val r = Random(100)
        val shape = 0.1f
        val s = generateSequence { r.nextGamma(shape) }
                .take(300)
                .map { assertFalse(it <= 0, "$it"); it }
                .sample(RunningVariance())
        assertEquals(shape, s.variance, 0.2f)
        assertEquals(shape, s.mean, 0.1f)
    }

    @Test
    fun gammaShapeEquals1() {
        val r = Random(2934)
        val shape = 1.0f
        val s = generateSequence { r.nextGamma(shape) }
                .take(200)
                .map { assertFalse(it <= 0, "$it"); it }
                .sample(RunningVariance())
        assertEquals(shape, s.variance, 0.2f)
        assertEquals(shape, s.mean, 0.1f)
    }

    @Test
    fun gammaShapeGreaterThan1() {
        val r = Random(12934)
        val shape = 2.0f
        val s = generateSequence { r.nextGamma(shape) }
                .take(200)
                .map { assertFalse(it <= 0, "$it"); it }
                .sample(RunningVariance())
        assertEquals(shape, s.variance, 0.25f)
        assertEquals(shape, s.mean, 0.2f)
    }

    @Test
    fun gammaShapeHuge() {
        val r = Random(1934)
        val shape = 100.0f
        val s = generateSequence { r.nextGamma(shape) }
                .take(200)
                .map { assertFalse(it <= 0, "$it"); it }
                .sample(RunningVariance())
        assertEquals(shape, s.variance, 10.0f)
        assertEquals(shape, s.mean, 5.0f)
    }

    @Test
    fun betaUniform() {
        val r = Random(1023)
        val s = generateSequence { r.nextBeta(1.0f, 1.0f) }
                .take(200)
                .map { assertFalse(it <= 0, "$it"); assertFalse(it >= 1, "$it"); it }
                .sample(RunningVariance())
        assertEquals(1 / 12.0f, s.variance, 0.1f)
        assertEquals(0.5f, s.mean, 0.1f)
    }

    @Test
    fun betaSkewedDown() {
        val r = Random(12410)
        val a = 1.0f
        val b = 10.0f
        val s = generateSequence { r.nextBeta(a, b) }
                .take(200)
                .map { assertFalse(it <= 0, "$it"); assertFalse(it >= 1, "$it"); it }
                .sample(RunningVariance())
        assertEquals((a * b) / ((a + b).pow(2) * (a + b + 1)), s.variance, 0.1f)
        assertEquals(a / (a + b), s.mean, 0.1f)
    }

    @Test
    fun betaSkewedUp() {
        val r = Random(-10)
        val a = 10.0f
        val b = 1.0f
        val s = generateSequence { r.nextBeta(a, b) }
                .take(200)
                .map { assertFalse(it <= 0, "$it"); assertFalse(it >= 1, "$it"); it }
                .sample(RunningVariance())
        assertEquals((a * b) / ((a + b).pow(2) * (a + b + 1)), s.variance, 0.1f)
        assertEquals(a / (a + b), s.mean, 0.1f)
    }

    @Test
    fun poissonSmall() {
        val r = Random(548)
        val lambda = 0.1f
        val s = generateSequence { r.nextPoisson(lambda) }
                .take(200)
                .map { assertFalse(it < 0, "$it"); it }
                .sample(RunningVariance())
        assertEquals(lambda, s.variance, 0.1f)
        assertEquals(lambda, s.mean, 0.05f)
    }

    @Test
    fun poissonMedium() {
        val r = Random(1546)
        val lambda = 2.0f
        val s = generateSequence { r.nextPoisson(lambda) }
                .take(300)
                .map { assertFalse(it < 0, "$it"); it }
                .sample(RunningVariance())
        assertEquals(lambda, s.variance, 0.5f)
        assertEquals(lambda, s.mean, 0.2f)
    }

    @Test
    fun poissonHuge() {
        val r = Random(68)
        val lambda = 100.0f // will trigger normal approximation
        val s = generateSequence { r.nextPoisson(lambda) }
                .take(300)
                .map { assertFalse(it < 0, "$it"); it }
                .sample(RunningVariance())
        assertEquals(lambda, s.variance, 25.0f)
        assertEquals(lambda, s.mean, 1.0f)
    }

    @Test
    fun bernoulli() {
        val r = Random(568)
        val p = 0.1f
        val s = generateSequence { r.nextBinomial(p) }
                .take(200)
                .map { assertFalse(it < 0, "$it"); assertFalse(it > 1, "$it"); it }
                .sample(RunningVariance())
        assertEquals(p * (1 - p), s.variance, 0.2f)
        assertEquals(p, s.mean, 0.1f)
    }

    @Test
    fun binomialHuge() {
        val r = Random(789)
        val p = 0.2f
        val n = 100
        val s = generateSequence { r.nextBinomial(p, n) }
                .take(200)
                .map { assertFalse(it < 0, "$it"); it }
                .sample(RunningVariance())
        assertEquals(n * p * (1 - p), s.variance, 10.0f)
        assertEquals(n * p, s.mean, 5.0f)
    }

    @Test
    fun geometric() {
        val r = Random(978546)
        val p = 10.0f
        val s = generateSequence { r.nextGeometric(p) }
                .take(200)
                .map { assertFalse(it < 0, "$it"); it }
                .sample(RunningVariance())
        assertEquals((1.0f - p) / p.pow(2), s.variance, 10.0f)
        assertEquals(1.0f / p, s.mean, 5.0f)
    }

    @Test
    fun exponential() {
        val r = Random(100)
        val rate = 10.0f
        val s = generateSequence { r.nextExponential(rate) }
                .take(200)
                .map { assertFalse(it < 0, "$it"); it }
                .sample(RunningVariance())
        assertEquals(1.0f / rate.pow(2), s.variance, 0.5f)
        assertEquals(1.0f / rate, s.mean, 0.1f)
    }
}
