package combo.util

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse

class BoundedBlockSinkTest {
    @Test
    fun offerRejectsOverflow() {
        val s = BlockingSink<Int>(10)
        for (i in 1..10)
            s.add(i)
        assertFalse(s.offer(0))
    }

    @Test
    fun drainConsumes() {
        val s = BlockingSink<Int>(10)
        for (i in 1..10)
            s.add(i)
        assertEquals(10, s.drain().toList().size)
        assertEquals(0, s.drain().toList().size)
    }

    @Test
    fun addAfterDrain() {
        val s = BlockingSink<Int>(10)
        for (i in 1..10)
            s.add(i)
        assertEquals(10, s.drain().toList().size)
        s.add(0)
        assertEquals(1, s.drain().toList().size)
    }
}
