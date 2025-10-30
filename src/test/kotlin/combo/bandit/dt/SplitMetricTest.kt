package combo.bandit.dt

import combo.math.RunningVariance
import combo.math.VarianceEstimator
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SplitMetricTest {
    val total: VarianceEstimator
    val pos: Array<VarianceEstimator>
    val neg: Array<VarianceEstimator>

    init {
        val posData1 = FloatArray(17) { 1.0f }
        val negData1 = FloatArray(15) { 0.0f }
        posData1[4] = 0.0f
        posData1[10] = 0.0f
        negData1[2] = 1.0f
        negData1[8] = 1.0f
        negData1[10] = 1.0f

        val posData2 = FloatArray(12) { 0.0f }
        for (i in 0 until 8) {
            posData2[i] = 1.0f
        }
        val negData2 = FloatArray(20) { 0.0f }
        for (i in 0 until 9) {
            negData2[i] = 1.0f
        }

        total = RunningVariance()
        total.acceptAll(posData1)
        total.acceptAll(negData1)
        val total2 = RunningVariance()
        total2.acceptAll(posData2)
        total2.acceptAll(negData2)

        val pos1 = RunningVariance()
        pos1.acceptAll(posData1)
        val pos2 = RunningVariance()
        pos2.acceptAll(posData2)
        val neg1 = RunningVariance()
        neg1.acceptAll(negData1)
        val neg2 = RunningVariance()
        neg2.acceptAll(negData2)

        pos = arrayOf(pos1, pos2)
        neg = arrayOf(neg1, neg2)
    }

    @Test
    fun varianceReduction() {
        val sm = VarianceReduction
        val split = sm.split(total, pos, neg, 1.0f, 1.0f)
        assertEquals(0, split.i)
        assertTrue(split.top1 > split.top2)
    }

    @Test
    fun fTest() {
        val sm = TTest
        val split = sm.split(total, pos, neg, 1.0f, 1.0f)
        assertEquals(0, split.i)
        assertTrue(split.top1 > split.top2)
    }

    @Test
    fun chi2Test() {
        val sm = ChiSquareTest
        val split = sm.split(total, pos, neg, 1.0f, 1.0f)
        assertEquals(0, split.i)
        assertTrue(split.top1 > split.top2)
    }

    @Test
    fun totalEntropy() {
        val sm = EntropyReduction
        val rv = RunningVariance()
        rv.accept(16.0f / 30.0f, 30.0f)
        assertEquals(0.99f, sm.totalValue(rv), 0.01f)
    }

    @Test
    fun entropyValue() {
        val sm = EntropyReduction
        val pos = RunningVariance()
        pos.accept(12.0f / 30.0f, 30.0f)
        val neg = RunningVariance()
        neg.accept(4.0f / 30.0f, 30.0f)
        assertEquals(0.39f, sm.value(total, pos, neg), 0.79f)
    }

    @Test
    fun entropyRecuction() {
        val sm = EntropyReduction
        val split = sm.split(total, pos, neg, 1.0f, 1.0f)
        assertEquals(0, split.i)
        assertTrue(split.top1 > split.top2)
    }

    @Test
    fun giniCoefficient() {
        val sm = GiniCoefficient
        val split = sm.split(total, pos, neg, 1.0f, 1.0f)
        assertEquals(0, split.i)
        assertTrue(split.top1 > split.top2)
    }
}
