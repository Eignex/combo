package combo.math

import combo.test.assertEquals
import combo.util.mapArray
import combo.util.removeAt
import combo.util.sumByFloat
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.*

class VarianceCombinationTest {

    @Test
    fun combineEmptyVariances() {
        assertFalse(combineVariance(0f, 0f, 0f, 0f, 0f, 0f).isNaN())
        assertFalse(combineVariance(0f, 0f, 0f, 0f, 1f, 1f).isNaN())
        assertFalse(combinePrecision(0f, 0f, 0f, 0f, 0f, 0f).isNaN())
        assertFalse(combinePrecision(0f, 0f, 0f, 0f, 1f, 1f).isNaN())
    }

    @Test
    fun combineVariances() {
        val x1 = floatArrayOf(1f, -2f, 3f, 4f)
        val x2 = floatArrayOf(-1f, -1f, 0f, 3f, 5f, 1f, 2f, -3f, 3f)
        val m1 = x1.average().toFloat()
        val m2 = x2.average().toFloat()
        val n1 = x1.size.toFloat()
        val n2 = x2.size.toFloat()

        val v1 = 7f
        val v2 = 5.5555553f
        val v = combineVariance(v1, v2, m1, m2, n1, n2)
        val p = combinePrecision(1f / v1, 1 / v2, m1, m2, n1, n2)
        assertEquals(6.05325f, v, 1e-4f)
        assertEquals(v, 1 / p, 1e-6f)
    }
}

class RunningVarianceTest {

    @Test
    fun fixedSamples() {
        val s = RunningVariance()
        val values = floatArrayOf(2.1f, 2.4f, 3.3f, 2.5f, 2.2f, 4.1f, 3.9f, 2.8f)
        val mean = values.sum() / values.size
        for (value in values)
            s.accept(value)
        assertEquals(mean, s.mean, 0.1f)
        val variance = values.sumByFloat { (it - mean).pow(2) } / values.size
        assertEquals(variance, s.variance, 0.2f)
    }

    @Test
    fun fixedSamplesRemove() {
        val values = floatArrayOf(2.1f, 2.4f, 3.3f, 2.5f, 2.2f, 4.1f, 3.9f, 2.8f)
        val v1 = values.asSequence().sample(RunningVariance())
        v1.remove(2.1f)
        val v2 = values.slice(1 until values.size).asSequence().sample(RunningVariance())
        assertEquals(v1.mean, v2.mean, 0.001f)
        assertEquals(v1.variance, v2.variance, 0.001f)
        assertEquals(v1.nbrWeightedSamples, v2.nbrWeightedSamples, 0.001f)
    }

    @Test
    fun randomSamples() {
        val r = Random(100)
        val s = generateSequence { r.nextNormal(3.0f, sqrt(4.0f)) }.take(200).sample(RunningVariance())
        assertEquals(3.0f, s.mean, 0.2f)
        assertEquals(4.0f, s.variance, 0.5f)
        assertEquals(2.0f, s.standardDeviation, 0.25f)
    }

    @Test
    fun fixedWeightedSamples() {
        val rv = RunningVariance()
        val values = floatArrayOf(4.3f, -0.4f, 3.5f, 2.5f, 5.4f, -0.1f, 3.0f, 2.2f)
        val weights = floatArrayOf(2.0f, 2.3f, 2.3f, 0.9f, 1.7f, 1.2f, 0.8f, 2.0f)
        for ((i, v) in values.withIndex())
            rv.accept(v, weights[i])
        val sum = FloatVector(values) dot FloatVector(weights)
        val mean = sum / weights.sum()
        assertEquals(rv.nbrWeightedSamples, weights.sum(), 0.1f)
        assertEquals(mean, rv.mean, 0.1f)
        val variance = values.mapIndexed { i, v ->
            weights[i] * (v - mean).pow(2)
        }.sum() / weights.sum()
        assertEquals(variance, rv.variance, 0.1f)
    }

    @Test
    fun fixedWeightedSamplesRemove() {
        val v1 = RunningVariance()
        val values = floatArrayOf(4.3f, -0.4f, 3.5f, 2.5f, 5.4f, -0.1f, 3.0f, 2.2f)
        val weights = floatArrayOf(2.0f, 2.3f, 2.3f, 0.9f, 1.7f, 1.2f, 0.8f, 2.0f)
        for ((i, v) in values.withIndex())
            v1.accept(v, weights[i])
        v1.remove(3.5f, 2.3f)

        val v2 = RunningVariance()
        val values2 = values.removeAt(2)
        val weights2 = weights.removeAt(2)
        for ((i, v) in values2.withIndex())
            v2.accept(v, weights2[i])

        assertEquals(v2.mean, v1.mean, 0.001f)
        assertEquals(v2.variance, v1.variance, 0.001f)
        assertEquals(v2.nbrWeightedSamples, v1.nbrWeightedSamples, 0.001f)
    }

    @Test
    fun equalsTests() {
        val v1 = RunningVariance()
        val v2 = RunningVariance()
        assertFalse(v1.equals(RunningMean()))
        v1.accept(1.0f)
        assertNotEquals(v1, v2)
        v2.accept(1.0f)
        assertEquals(v1, v2)
        assertFalse(v1.equals(null))
    }

    @Test
    fun hashCodeTests() {
        val v1 = RunningVariance()
        val v2 = RunningVariance()
        assertEquals(v1.hashCode(), v2.hashCode())
        v1.accept(1.0f)
        v2.accept(1.0f)
        assertEquals(v1.hashCode(), v2.hashCode())
    }

    @Test
    fun combineEmptyTest() {
        val v1 = RunningVariance()
        val v2 = RunningVariance()
        val v = v1.combine(v2)
        assertFalse(v.mean.isNaN())
        assertTrue(v.variance.isNaN())
        assertFalse(v.nbrWeightedSamples.isNaN())
        assertEquals(v, v1)
        assertEquals(v, v2)
    }
}

class RunningMeanTest {

    @Test
    fun fixedSamples() {
        val s = RunningMean()
        val values = floatArrayOf(2.0f, 3.0f, 0.0f, 4.0f, 3.0f, 4.0f, 4.0f, 3.0f)
        val mean = values.sum() / values.size
        for (value in values)
            s.accept(value)
        assertEquals(mean, s.mean, 0.1f)
        assertEquals(mean, s.variance, 0.1f)
    }

    @Test
    fun fixedSamplesRemove() {
        val values = floatArrayOf(2.1f, 2.4f, 3.3f, 2.5f, 2.2f, 4.1f, 3.9f, 2.8f)
        val v1 = values.asSequence().sample(RunningMean())
        v1.remove(2.1f, 1.0f)
        val v2 = values.slice(1 until values.size).asSequence().sample(RunningMean())
        assertEquals(v1.mean, v2.mean, 0.001f)
        assertEquals(v1.nbrWeightedSamples, v2.nbrWeightedSamples, 0.001f)
    }

    @Test
    fun randomSamples() {
        val r = Random(100)
        val s = generateSequence { r.nextPoisson(10.0f) }.take(200).sample(RunningMean())
        assertEquals(10.0f, s.mean, 0.3f)
        assertEquals(10.0f, s.variance, 0.3f)
        assertEquals(sqrt(10.0f), s.standardDeviation, 0.1f)
    }

    @Test
    fun fixedWeightedSamples() {
        val rv = RunningMean()
        val values = floatArrayOf(4.3f, -0.4f, 3.5f, 2.5f, 5.4f, -0.1f, 3.0f, 2.2f)
        val weights = floatArrayOf(2.0f, 2.3f, 2.3f, 0.9f, 1.7f, 1.2f, 0.8f, 2.0f)
        for ((i, v) in values.withIndex())
            rv.accept(v, weights[i])
        val sum = FloatVector(values) dot FloatVector(weights)
        val mean = sum / weights.sum()
        assertEquals(rv.nbrWeightedSamples, weights.sum(), 0.1f)
        assertEquals(mean, rv.mean, 0.1f)
    }

    @Test
    fun fixedWeightedSamplesRemove() {
        val v1 = RunningMean()
        val values = floatArrayOf(4.3f, -0.4f, 3.5f, 2.5f, 5.4f, -0.1f, 3.0f, 2.2f)
        val weights = floatArrayOf(2.0f, 2.3f, 2.3f, 0.9f, 1.7f, 1.2f, 0.8f, 2.0f)
        for ((i, v) in values.withIndex())
            v1.accept(v, weights[i])
        v1.remove(3.5f, 2.3f)

        val v2 = RunningMean()
        val values2 = values.removeAt(2)
        val weights2 = weights.removeAt(2)
        for ((i, v) in values2.withIndex())
            v2.accept(v, weights2[i])

        assertEquals(v2.mean, v1.mean, 0.001f)
        assertEquals(v2.nbrWeightedSamples, v1.nbrWeightedSamples, 0.001f)
    }

    @Test
    fun equalsTests() {
        val v1 = RunningMean()
        val v2 = RunningMean()
        assertFalse(v1.equals(BinarySum()))
        v1.accept(1.0f)
        assertNotEquals(v1, v2)
        v2.accept(1.0f)
        assertEquals(v1, v2)
        assertFalse(v1.equals(null))
    }

    @Test
    fun hashCodeTests() {
        val v1 = RunningMean()
        val v2 = RunningMean()
        assertEquals(v1.hashCode(), v2.hashCode())
        v1.accept(1.0f)
        v2.accept(1.0f)
        assertEquals(v1.hashCode(), v2.hashCode())
    }

    @Test
    fun combineEmptyTest() {
        val v1 = RunningMean()
        val v2 = RunningMean()
        val v = v1.combine(v2)
        assertFalse(v.mean.isNaN())
        assertFalse(v.variance.isNaN())
        assertFalse(v.nbrWeightedSamples.isNaN())
        assertEquals(v, v1)
        assertEquals(v, v2)
    }
}

class ExponentialDecayVarianceTest {

    @Test
    fun fixedSamples() {
        val s = ExponentialDecayVariance(8)
        val values = floatArrayOf(2.1f, 2.4f, 3.3f, 2.5f, 2.2f, 4.1f, 3.9f, 2.8f)
        val mean = values.sum() / values.size
        for (value in values)
            s.accept(value)
        assertEquals(mean, s.mean, 0.1f)
        val variance = values.sumByFloat { (it - mean).pow(2) } / values.size
        assertEquals(variance, s.variance, 0.1f)
    }

    @Test
    fun randomSamples() {
        val r = Random(100)
        val s = generateSequence { r.nextNormal(3.0f, sqrt(4.0f)) }.take(200).sample(ExponentialDecayVariance(50))
        assertEquals(3.0f, s.mean, 0.1f)
        assertEquals(4.0f, s.variance, 0.1f)
        assertEquals(2.0f, s.standardDeviation, 0.1f)
    }

    @Test
    fun randomWeightedSamples() {
        val r = Random(100)
        val s = ExponentialDecayVariance(500)
        generateSequence { r.nextNormal(3.0f, sqrt(4.0f)) }.take(200).forEach { s.accept(it, 10.0f) }
        assertEquals(3.0f, s.mean, 0.1f)
        assertEquals(4.0f, s.variance, 0.1f)
        assertEquals(2.0f, s.standardDeviation, 0.1f)
    }

    @Test
    fun fixedWeightedSamples() {
        val values = floatArrayOf(4.3f, -0.4f, 3.5f, 2.5f, 5.4f, -0.1f, 3.0f, 2.2f)
        val weights = floatArrayOf(2.0f, 2.3f, 2.3f, 0.9f, 1.7f, 1.2f, 0.8f, 2.0f).mapArray { it * 20 }
        val edv = ExponentialDecayVariance(weights.sum().roundToInt())
        for ((i, v) in values.withIndex())
            edv.accept(v, weights[i])
        val sum = FloatVector(values) dot FloatVector(weights)
        val mean = sum / weights.sum()
        assertEquals(edv.nbrWeightedSamples, weights.sum(), 1E-6f)
        assertEquals(mean, edv.mean, 0.5f)
        val variance = values.mapIndexed { i, v ->
            weights[i] * (v - mean).pow(2)
        }.sum() / weights.sum()
        assertEquals(variance, edv.variance, 1.0f)
    }

    @Test
    fun equalsTests() {
        val v1 = ExponentialDecayVariance()
        val v2 = ExponentialDecayVariance()
        assertFalse(v1.equals(RunningVariance()))
        v1.accept(1.0f)
        assertNotEquals(v1, v2)
        v2.accept(1.0f)
        assertEquals(v1, v2)
        assertFalse(v1.equals(null))
    }

    @Test
    fun hashCodeTests() {
        val v1 = ExponentialDecayVariance()
        val v2 = ExponentialDecayVariance()
        assertEquals(v1.hashCode(), v2.hashCode())
        v1.accept(1.0f)
        v2.accept(1.0f)
        assertEquals(v1.hashCode(), v2.hashCode())
    }

    @Test
    fun combineEmptyTest() {
        val v1 = ExponentialDecayVariance()
        val v2 = ExponentialDecayVariance()
        val v = v1.combine(v2)
        assertFalse(v.mean.isNaN())
        assertFalse(v.variance.isNaN())
        assertFalse(v.nbrWeightedSamples.isNaN())
        assertEquals(v, v1)
        assertEquals(v, v2)
    }
}

class BinarySumTest {
    @Test
    fun fixedSamples() {
        val s = BinarySum()
        for (value in floatArrayOf(1.0f, 0.3f, 0.2f, 0.1f))
            s.accept(value)
        assertEquals(0.4f, s.mean, 1E-6f)
        assertEquals(0.4f * 0.6f, s.variance, 1E-6f)
    }

    @Test
    fun fixedSamplesRemove() {
        val values = floatArrayOf(1.0f, 0.3f, 0.2f, 0.1f)
        val v1 = values.asSequence().sample(BinarySum())
        v1.remove(0.3f, 1.0f)
        val v2 = values.removeAt(1).asSequence().sample(BinarySum())
        assertEquals(v1.mean, v2.mean, 0.001f)
        assertEquals(v1.variance, v2.variance, 0.001f)
        assertEquals(v1.nbrWeightedSamples, v2.nbrWeightedSamples)
    }

    @Test
    fun randomSamples() {
        val r = Random(101)
        val s = generateSequence { r.nextFloat() }.take(200).sample(BinarySum())
        assertEquals(0.5f, s.mean, 0.1f)
        assertEquals(0.25f, s.variance, 0.1f)
    }

    @Test
    fun fixedWeightedSamples() {
        val cd = BinarySum()
        val values = floatArrayOf(2.0f, 0.4f, 2.3f, 1.5f, 1.4f, 0.1f, 0.8f, 0.2f)
        val weights = floatArrayOf(2.0f, 2.3f, 2.9f, 2.9f, 1.7f, 1.2f, 3.8f, 2.0f)
        for ((i, v) in values.withIndex())
            cd.accept(v / weights[i], weights[i])
        val sum = values.sum()
        assertEquals(sum, cd.sum, 1E-6f)
        assertEquals(cd.nbrWeightedSamples, weights.sum(), 1E-6f)
        assertEquals(cd.mean, sum / cd.nbrWeightedSamples, 1E-6f)
    }

    @Test
    fun fixedWeightedSamplesRemove() {
        val v1 = RunningVariance()
        val values = floatArrayOf(2.0f, 0.4f, 2.3f, 1.5f, 1.4f, 0.1f, 0.8f, 0.2f)
        val weights = floatArrayOf(2.0f, 2.3f, 2.9f, 2.9f, 1.7f, 1.2f, 3.8f, 2.0f)
        for ((i, v) in values.withIndex())
            v1.accept(v, weights[i])
        v1.remove(2.3f, 2.9f)

        val v2 = RunningVariance()
        val values2 = values.removeAt(2)
        val weights2 = weights.removeAt(2)
        for ((i, v) in values2.withIndex())
            v2.accept(v, weights2[i])

        assertEquals(v2.mean, v1.mean, 0.001f)
        assertEquals(v2.variance, v1.variance, 0.001f)
        assertEquals(v2.nbrWeightedSamples, v1.nbrWeightedSamples, 0.001f)
    }

    @Test
    fun illegalValues() {
        assertFailsWith(IllegalArgumentException::class) {
            BinarySum().accept(5.0f, 2.0f)
        }
        assertFailsWith(IllegalArgumentException::class) {
            BinarySum().accept(-1.0f, 1.0f)
        }
    }

    @Test
    fun equalsTests() {
        val v1 = BinarySum()
        val v2 = BinarySum()
        assertFalse(v1.equals(RunningMean()))
        v1.accept(1.0f)
        assertNotEquals(v1, v2)
        v2.accept(1.0f)
        assertEquals(v1, v2)
        assertFalse(v1.equals(null))
    }

    @Test
    fun hashCodeTests() {
        val v1 = BinarySum()
        val v2 = BinarySum()
        assertEquals(v1.hashCode(), v2.hashCode())
        v1.accept(1.0f)
        v2.accept(1.0f)
        assertEquals(v1.hashCode(), v2.hashCode())
    }

    @Test
    fun combineEmptyTest() {
        val v1 = BinarySum()
        val v2 = BinarySum()
        val v = v1.combine(v2)
        assertTrue(v.mean.isNaN())
        assertTrue(v.variance.isNaN())
        assertFalse(v.nbrWeightedSamples.isNaN())
        assertEquals(v, v1)
        assertEquals(v, v2)
    }
}

class RunningSquaredMeansTest {

    @Test
    fun fixedSamples() {
        val s = RunningSquaredMeans()
        val values = floatArrayOf(2.1f, 2.4f, 3.3f, 2.5f, 2.2f, 4.1f, 3.9f, 2.8f)
        val mean = values.sum() / values.size
        for (value in values)
            s.accept(value)
        assertEquals(mean, s.mean, 0.1f)
        val variance = values.sumByFloat { (it - mean).pow(2) } / values.size
        assertEquals(variance, s.variance, 0.2f)
        val squaredMeans = values.map { it * it }.average()
        assertEquals(squaredMeans.toFloat(), s.meanOfSquares, 0.001f)
    }

    @Test
    fun fixedSamplesRemove() {
        val values = floatArrayOf(2.1f, 2.4f, 3.3f, 2.5f, 2.2f, 4.1f, 3.9f, 2.8f)
        val v1 = values.asSequence().sample(RunningSquaredMeans())
        v1.remove(2.1f)
        val v2 = values.slice(1 until values.size).asSequence().sample(RunningSquaredMeans())
        assertEquals(v1.mean, v2.mean, 0.001f)
        assertEquals(v1.variance, v2.variance, 0.001f)
        assertEquals(v1.nbrWeightedSamples, v2.nbrWeightedSamples, 0.001f)
        assertEquals(v1.meanOfSquares, v2.meanOfSquares, 0.001f)
    }

    @Test
    fun randomSamples() {
        val r = Random(100)
        val values = generateSequence { r.nextNormal(3.0f, sqrt(4.0f)) }.take(200).toList()
        val s = values.asSequence().sample(RunningSquaredMeans())
        assertEquals(3.0f, s.mean, 0.2f)
        assertEquals(4.0f, s.variance, 0.5f)
        assertEquals(2.0f, s.standardDeviation, 0.25f)
        val meanOfSquares = values.map { it * it }.average().toFloat()
        assertEquals(meanOfSquares, s.meanOfSquares, 0.5f)
    }

    @Test
    fun fixedWeightedSamples() {
        val rv = RunningSquaredMeans()
        val values = floatArrayOf(4.3f, -0.4f, 3.5f, 2.5f, 5.4f, -0.1f, 3.0f, 2.2f)
        val weights = floatArrayOf(2.0f, 2.3f, 2.3f, 0.9f, 1.7f, 1.2f, 0.8f, 2.0f)
        for ((i, v) in values.withIndex())
            rv.accept(v, weights[i])
        val sum = FloatVector(values) dot FloatVector(weights)
        val mean = sum / weights.sum()
        assertEquals(rv.nbrWeightedSamples, weights.sum(), 0.1f)
        assertEquals(mean, rv.mean, 0.1f)
        val variance = values.mapIndexed { i, v ->
            weights[i] * (v - mean).pow(2)
        }.sum() / weights.sum()
        assertEquals(variance, rv.variance, 0.1f)

        var meanOfSquares = 0.0f
        for (i in values.indices) {
            meanOfSquares += weights[i] * values[i] * values[i]
        }
        meanOfSquares /= weights.sum()
        assertEquals(meanOfSquares, rv.meanOfSquares, 0.1f)
    }

    @Test
    fun fixedWeightedSamplesRemove() {
        val v1 = RunningSquaredMeans()
        val values = floatArrayOf(4.3f, -0.4f, 3.5f, 2.5f, 5.4f, -0.1f, 3.0f, 2.2f)
        val weights = floatArrayOf(2.0f, 2.3f, 2.3f, 0.9f, 1.7f, 1.2f, 0.8f, 2.0f)
        for ((i, v) in values.withIndex())
            v1.accept(v, weights[i])
        v1.remove(3.5f, 2.3f)

        val v2 = RunningSquaredMeans()
        val values2 = values.removeAt(2)
        val weights2 = weights.removeAt(2)
        for ((i, v) in values2.withIndex())
            v2.accept(v, weights2[i])

        assertEquals(v2.mean, v1.mean, 0.001f)
        assertEquals(v2.variance, v1.variance, 0.001f)
        assertEquals(v2.nbrWeightedSamples, v1.nbrWeightedSamples, 0.001f)
    }

    @Test
    fun equalsTests() {
        val v1 = RunningSquaredMeans()
        val v2 = RunningSquaredMeans()
        assertFalse(v1.equals(RunningVariance()))
        v1.accept(1.0f)
        assertNotEquals(v1, v2)
        v2.accept(1.0f)
        assertEquals(v1, v2)
        assertFalse(v1.equals(null))
    }

    @Test
    fun hashCodeTests() {
        val v1 = RunningSquaredMeans()
        val v2 = RunningSquaredMeans()
        assertEquals(v1.hashCode(), v2.hashCode())
        v1.accept(1.0f)
        v2.accept(1.0f)
        assertEquals(v1.hashCode(), v2.hashCode())
    }

    @Test
    fun combine() {
        val rng = Random(0)
        val data1 = generateSequence { rng.nextNormal(2.0f, 4.0f) }.take(100).toList().toTypedArray()
        val data2 = generateSequence { rng.nextNormal(3.0f, 9.0f) }.take(20).toList().toTypedArray()
        val se1 = data1.asSequence().sample(RunningSquaredMeans())
        val se2 = data2.asSequence().sample(RunningSquaredMeans())
        val totalSeExpected = (data1 + data2).toList().shuffled().asSequence().sample(RunningSquaredMeans())
        val totalSeAcutal = se1.combine(se2)
        assertEquals(totalSeExpected.meanOfSquares, totalSeAcutal.meanOfSquares, 5.0f)
    }

    @Test
    fun combineEmptyTest() {
        val v1 = RunningSquaredMeans()
        val v2 = RunningSquaredMeans()
        val v = v1.combine(v2)
        assertFalse(v.mean.isNaN())
        assertTrue(v.variance.isNaN())
        assertFalse(v.nbrWeightedSamples.isNaN())
        assertEquals(v, v1)
        assertEquals(v, v2)
    }
}

class WindowedEstimatorTest {
    @Test
    fun one() {
        val v = WindowedEstimator(10, RunningVariance())
        v.accept(3.0f, 2.0f)
        val v2 = RunningVariance()
        v2.accept(3.0f, 2.0f)
        assertEquals(v.base, v2)
    }

    @Test
    fun overflow() {
        val v = WindowedEstimator(3, RunningVariance())
        val values = floatArrayOf(4.3f, -0.4f, 3.5f, 2.5f, 5.4f, -0.1f, 3.0f, 2.2f)
        val weights = floatArrayOf(2.0f, 2.3f, 2.3f, 0.9f, 1.7f, 1.2f, 0.8f, 2.0f)
        for (i in values.indices)
            v.accept(values[i], weights[i])
        val v2 = RunningVariance()
        for (i in 5 until 8)
            v2.accept(values[i], weights[i])
        assertEquals(v.mean, v2.mean, 0.001f)
        assertEquals(v.variance, v2.variance, 0.001f)
        assertEquals(v.nbrWeightedSamples, v2.nbrWeightedSamples, 0.001f)
    }
}

class WindowedSquaredEstimatorTest {
    @Test
    fun one() {
        val v = WindowedSquaredEstimator(10, RunningSquaredMeans())
        v.accept(3.0f, 2.0f)
        val v2 = RunningSquaredMeans()
        v2.accept(3.0f, 2.0f)
        assertEquals(v.base, v2)
    }

    @Test
    fun overflow() {
        val v = WindowedSquaredEstimator(3, RunningSquaredMeans())
        val values = floatArrayOf(4.3f, -0.4f, 3.5f, 2.5f, 5.4f, -0.1f, 3.0f, 2.2f)
        val weights = floatArrayOf(2.0f, 2.3f, 2.3f, 0.9f, 1.7f, 1.2f, 0.8f, 2.0f)
        for (i in values.indices)
            v.accept(values[i], weights[i])
        val v2 = RunningSquaredMeans()
        for (i in 5 until 8)
            v2.accept(values[i], weights[i])
        assertEquals(v.mean, v2.mean, 0.001f)
        assertEquals(v.variance, v2.variance, 0.001f)
        assertEquals(v.nbrWeightedSamples, v2.nbrWeightedSamples, 0.001f)
        assertEquals(v.meanOfSquares, v2.meanOfSquares, 0.001f)
    }
}
