package combo.bandit.univariate

import combo.math.*
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class UnivariatePosteriorsTest {

    @Test
    fun binomialUpdate() {
        val stat = BinomialPosterior.defaultPrior()
        val r = Random(1024)
        val s1 = generateSequence { BinomialPosterior.sample(stat, r) }.take(100).sample(RunningVariance())
        assertEquals(.5f, s1.mean, 0.2f)
        assertEquals(50.0f, s1.sum, 7.5f)
        floatArrayOf(1.0f, 1.0f, 0.0f).forEach { BinomialPosterior.update(stat, it) }
        val s2 = generateSequence { BinomialPosterior.sample(stat, r) }.take(100).sample(RunningVariance())
        assertEquals(3 / 5.0f, s2.mean, 0.2f)
        assertTrue(s2.variance < s1.variance)
    }

    @Test
    fun poissonUpdate() {
        val p = PoissonPosterior
        val stat = p.defaultPrior()
        val r = Random(19)
        val s1 = generateSequence { p.sample(stat, r) }.take(100).sample(RunningVariance())
        assertEquals(1.0f, s1.mean, 5.0f) // posterior has variance=100, so prior estimate is bad
        floatArrayOf(3.0f, 4.0f, 0.0f, 5.0f, 0.0f, 1.0f, 0.0f).forEach { p.update(stat, it) }
        val s2 = generateSequence { p.sample(stat, r) }.take(100).sample(RunningVariance())
        assertEquals(stat.mean, s2.mean, 0.5f)
        assertTrue(s2.variance < s1.variance)
    }

    @Test
    fun geometricUpdate() {
        val p = GeometricPosterior
        val stat = p.defaultPrior()
        val r = Random(46978)
        val s1 = generateSequence { p.sample(stat, r) }.take(100).sample(RunningVariance())
        assertEquals(0.5f, s1.mean, 0.1f)
        floatArrayOf(3.0f, 4.0f, 1.0f, 3.0f, 6.0f, 2.0f, 3.0f).forEach { p.update(stat, it) }
        val s2 = generateSequence { p.sample(stat, r) }.take(100).sample(RunningVariance())
        assertEquals(1 / stat.mean, s2.mean, 0.2f)
        assertTrue(s2.variance < s1.variance)
    }

    @Test
    fun normalPriorSignUnbiased() {
        val p = NormalPosterior
        val stat = p.defaultPrior()
        val r = Random(893)
        val dataSample = generateSequence { p.sample(stat, r) }.take(100).sample(FullSample())
        val neg = dataSample.values().count { it < 0 }
        val pos = dataSample.values().count { it > 0 }
        assertEquals(neg.toFloat(), pos.toFloat(), 30.0f)
    }

    @Test
    fun normalUpdate() {

        fun test(data: FloatArray, rng: Random) {
            // Ignoring the priors here
            val v = RunningVariance()
            data.forEach { v.accept(it) }
            val s = generateSequence { NormalPosterior.sample(v, rng) }.take(100).sample(RunningVariance())
            assertEquals(v.mean, s.mean, 0.1f)

            val alpha = v.nbrWeightedSamples / 2.0f
            val beta = v.squaredDeviations / 2.0f
            val lambda = v.nbrWeightedSamples

            assertEquals(s.variance, (beta / (alpha - 1)) / lambda, 1.0f / lambda)
        }
        test(floatArrayOf(1.0f, 2.0f, 3.0f), Random(0))
        test(floatArrayOf(3.0f, 4.0f, 1.0f, 3.0f, 6.0f, 2.0f, 2.0f), Random(0))
        test(generateSequence { Random.nextNormal() }.take(50).toList().toFloatArray(), Random(0))
    }

    @Test
    fun logNormalUpdate() {
        val p = LogNormalPosterior
        val stat = p.defaultPrior()
        val r = Random
        val mean = 5.0f
        val variance = 10.0f
        for (i in 1..1) p.update(stat, r.nextLogNormal(mean, variance))
        val v = RunningVariance()
        for (i in 1..1000) v.accept(p.sample(stat, r))
    }

    @Test
    fun exponentialUpdate() {
        val p = ExponentialPosterior
        val stat = p.defaultPrior()
        val r = Random(75)
        val s1 = generateSequence { p.sample(stat, r) }.take(50).sample(RunningVariance())
        floatArrayOf(3.0f, 4.0f, 1.0f, 3.0f, 6.0f, 2.0f, 2.0f).forEach { p.update(stat, it) }
        val s2 = generateSequence { p.sample(stat, r) }.take(50).sample(RunningVariance())
        assertEquals(1 / 3.0f, s2.mean, 0.1f)
        assertTrue(s2.variance < s1.variance)
    }

    @Test
    fun gammaScaleUpdate() {
        val p = GammaScalePosterior(fixedShape = 2.0f)
        val stat = p.defaultPrior()
        val r = Random(1023)
        val s1 = generateSequence { p.sample(stat, r) }.take(50).sample(RunningVariance())
        floatArrayOf(3.0f, 4.0f, 1.0f, 3.0f, 6.0f, 2.0f, 2.0f).forEach { p.update(stat, it) }
        val s2 = generateSequence { p.sample(stat, r) }.take(50).sample(RunningVariance())
        assertEquals(2 / 3.0f, s2.mean, 0.1f)
        assertTrue(s2.variance < s1.variance)
    }

    @Test
    fun properPriors() {
        for ((i, posterior) in arrayOf(PoissonPosterior, ExponentialPosterior, NormalPosterior, GammaScalePosterior(0.1f),
                BinomialPosterior, GeometricPosterior, LogNormalPosterior).withIndex()) {
            val rng = Random(i)
            val prior = posterior.defaultPrior()
            assertTrue(prior.variance.isFinite(), i.toString())
            assertTrue(prior.mean.isFinite(), i.toString())
            val s = RunningVariance()
            generateSequence { posterior.sample(prior, rng) }.take(50).forEach { s.accept(it) }
            assertTrue(s.variance.isFinite(), i.toString())
            assertTrue(s.variance > 0, i.toString())
            assertTrue(s.mean.isFinite(), i.toString())
        }
        for ((i, posterior) in arrayOf(NormalPosterior, LogNormalPosterior).withIndex()) {
            val prior = posterior.defaultPrior()
            assertTrue(prior.variance > 0, i.toString())
        }
    }

    @Test
    fun hierarchicalNormalShrinkage() {
        val groups = 8
        val dataPoints = 5
        val experiments = 2

        // This test varies the error of two sets of groups.
        // The estimated mean of the second group should be shrunk towards the pool mean because the error is larger
        // there.
        val weights = FloatArray(groups) { 1 + it.toFloat() }

        val rng = Random(0)
        val posteriors = Array(experiments) { HierarchicalNormalPosterior(PooledVarianceEstimator(RunningVariance(0.0f, 0.02f, 0.02f))) }
        val data = Array(experiments) { i ->
            val err = 10 * i.toFloat() + 0.5f
            val pool = posteriors[i].pool
            Array(weights.size) { j ->
                RunningVariance(1 + j.toFloat(), err * dataPoints, dataPoints.toFloat()).also { pool.addArm(it) }
            }.also { pool.recalculate() }
        }

        val samples = Array(experiments) { i ->
            Array(weights.size) { j ->
                generateSequence { posteriors[i].sample(data[i][j], rng) }.take(100).sample(RunningVariance())
            }
        }

        val means = samples.map { s -> s.map { it.mean }.toTypedArray().asSequence().sample(RunningVariance()) }
        assertEquals((groups + 1) / 2.0f, means[0].mean, 0.1f)
        assertEquals((groups + 1) / 2.0f, means[1].mean, 0.1f)
        assertTrue(means[0].variance > means[1].variance)
    }
}
