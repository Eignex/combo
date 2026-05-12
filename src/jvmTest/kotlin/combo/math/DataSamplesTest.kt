package combo.math

import combo.test.assertContentEquals
import kotlin.test.Test
import kotlin.test.assertEquals

class DataSamplesTest {

    @Test
    fun emptyGrowingSample() {
        assertEquals(0, BucketSample(10).values().size)
        assertEquals(0, BucketSample(10).labels().size)
    }

    @Test
    fun growingSample() {
        val s = BucketSample(10)
        for (i in 1..100)
            s.accept(i.toFloat())
        assertContentEquals(floatArrayOf(8.5f, 24.5f, 40.5f, 56.5f, 72.5f, 88.5f), s.values())
        assertContentEquals(longArrayOf(16, 32, 48, 64, 80, 96), s.labels())
    }

    @Test
    fun emptyFullSample() {
        assertEquals(0, FullSample().values().size)
    }

    @Test
    fun fullSample() {
        val s = FullSample()
        for (i in 1..100)
            s.accept(i.toFloat())
        assertContentEquals(FloatArray(100) { (1 + it).toFloat() }, s.values())
    }
}
