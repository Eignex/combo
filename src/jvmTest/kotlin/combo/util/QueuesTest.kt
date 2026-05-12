package combo.util

import kotlin.random.Random
import kotlin.test.*

class FloatCircleBufferTest {

    @Test
    fun addToSize() {
        val cq = FloatCircleBuffer(2)
        assertEquals(0, cq.size)
        cq.add(1.0f)
        assertEquals(1, cq.size)
        cq.add(2.0f)
        assertEquals(2, cq.size)
    }

    @Test
    fun addOverflow() {
        val aq = FloatCircleBuffer(2)
        aq.add(1.0f)
        aq.add(2.0f)
        aq.add(3.0f)
        assertFalse(aq.contains(1.0f))
        assertTrue(aq.contains(2.0f))
        assertTrue(aq.contains(3.0f))
    }

    @Test
    fun empty() {
        val cq = FloatCircleBuffer(10)
        assertFalse(cq.iterator().hasNext())
        assertFailsWith(NoSuchElementException::class) {
            cq.iterator().next()
        }
    }
}

class ArrayQueueTest {
    @Test
    fun addRemove() {
        val aq = ArrayQueue<Int>()
        aq.add(1)
        aq.add(2)
        aq.add(3)
        assertEquals(1, aq.poll())
        assertEquals(2, aq.poll())
        assertEquals(3, aq.poll())
        assertNull(aq.poll())
    }

    @Test
    fun addRemoveLots() {
        val aq = ArrayQueue<Int>()
        for (i in 0 until 10)
            aq.add(i)
        for (i in 0 until 10)
            assertEquals(i, aq.poll())

        aq.add(100)
        aq.add(200)
        assertEquals(100, aq.poll())
        aq.add(300)
        aq.add(400)
        assertEquals(200, aq.poll())
    }

    @Test
    fun pollEmpty() {
        val aq = ArrayQueue<String>()
        assertNull(aq.poll())
    }

    @Test
    fun addPollPollEmpty() {
        val aq = ArrayQueue<Int>()
        aq.add(1)
        aq.poll()
        assertNull(aq.poll())
        assertNull(aq.peek())
    }

    @Test
    fun iteratorEmpty() {
        val aq = ArrayQueue<String>()
        assertFalse(aq.iterator().hasNext())
        assertFailsWith(NoSuchElementException::class) {
            aq.iterator().next()
        }
    }

    @Test
    fun iterator() {
        val aq = ArrayQueue<Int>()
        for (i in 0 until 10)
            aq.add(i)
        for (i in 0 until 10)
            assertTrue(aq.contains(i))
        aq.poll()
        for (i in 1 until 10)
            assertTrue(aq.contains(i))
        assertFalse(aq.contains(0))
    }

    @Test
    fun iteratorWrapAround() {
        val aq = ArrayQueue<String>()
        aq.add("a")
        aq.add("b")
        aq.add("c")
        aq.add("d")
        // write = 4, read = 0

        aq.poll()
        aq.poll()
        aq.add("e")
        aq.add("f")
        aq.add("g")
        aq.add("h")

        // write = 0, read = 2
        assertTrue(aq.contains("c"))
        assertTrue(aq.contains("h"))
        assertFalse(aq.contains("a"))

        aq.add("i")
        aq.add("j")
        aq.add("k")
        aq.poll()
        assertEquals(listOf("d", "e", "f", "g", "h", "i", "j", "k"), aq.toList())
    }

    @Test
    fun randomTest() {
        val aq = ArrayQueue<Int>()
        val rng = Random
        var prev = -1
        for (i in 0 until 1000) {
            if (aq.size > 0 && rng.nextFloat() > 0.055f) {
                val poll = aq.poll()!!
                assertEquals(poll, prev + 1)
                prev = poll
            }
            aq.add(i)
        }
        prev = aq.peek()!!
        for (i in aq.iterator().asSequence().toList().subList(1, aq.size)) {
            assertEquals(i, prev + 1)
            prev = i
        }
    }
}