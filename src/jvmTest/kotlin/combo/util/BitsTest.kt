package combo.util

import kotlin.test.Test
import kotlin.test.assertEquals

class BitsTest {
    @Test
    fun bsr() {
        assertEquals(0, Int.bsr(0))
        assertEquals(0, Int.bsr(1))
        assertEquals(1, Int.bsr(2))
        assertEquals(1, Int.bsr(3))
        assertEquals(3, Int.bsr(10))
        assertEquals(5, Int.bsr(32))
        assertEquals(9, Int.bsr(1023))
        assertEquals(10, Int.bsr(1024))
        assertEquals(30, Int.bsr(Int.MAX_VALUE))
        assertEquals(31, Int.bsr(Int.MIN_VALUE))
        assertEquals(31, Int.bsr(-1))
        assertEquals(31, Int.bsr(-1241))
        assertEquals(31, Int.bsr(-104192))
    }

    @Test
    fun bsf() {
        assertEquals(0, Int.bsf(0))
        assertEquals(0, Int.bsf(1))
        assertEquals(1, Int.bsf(2))
        assertEquals(0, Int.bsf(3))
        assertEquals(1, Int.bsf(10))
        assertEquals(5, Int.bsf(32))
        assertEquals(0, Int.bsf(1023))
        assertEquals(10, Int.bsf(1024))
        assertEquals(0, Int.bsf(1025))
        assertEquals(0, Int.bsf(Int.MAX_VALUE))
        assertEquals(31, Int.bsf(Int.MIN_VALUE))
        assertEquals(0, Int.bsf(-1))
        assertEquals(3, Int.bsf(-1240))
        assertEquals(8, Int.bsf(-104192))
    }

    @Test
    fun floatMinValue() {
        val f1 = MIN_VALUE32
        val i = f1.toBits()
        val f2 = Float.fromBits(i)
        assertEquals(f1.toBits(), f2.toBits())
    }
}
