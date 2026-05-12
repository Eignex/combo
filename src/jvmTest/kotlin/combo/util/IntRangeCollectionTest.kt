package combo.util

import combo.test.assertContentEquals
import kotlin.random.Random
import kotlin.test.*

class IntRangeCollectionTest {

    @Test
    fun createPointSet() {
        val range = IntRangeCollection(0, 0)
        assertEquals(1, range.size)
        assertFalse(range.isEmpty())
        assertEquals(0, range[0])
    }

    @Test
    fun indexOf() {
        val range = IntRangeCollection(10, 12)
        assertEquals(-1, range.indexOf(9))
        assertEquals(0, range.indexOf(10))
        assertEquals(1, range.indexOf(11))
        assertEquals(2, range.indexOf(12))
        assertEquals(-1, range.indexOf(13))
    }

    @Test
    fun get() {
        val range = IntRangeCollection(10, 12)
        assertEquals(10, range[0])
        assertEquals(11, range[1])
        assertEquals(12, range[2])
    }

    @Test
    fun getOutOfBounds() {
        val range = IntRangeCollection(10, 12)
        assertFailsWith(IndexOutOfBoundsException::class) {
            range[-1]
        }
        assertFailsWith(IndexOutOfBoundsException::class) {
            range[3]
        }
    }

    @Test
    fun containsPointSet() {
        val range = IntRangeCollection(0, 0)
        assertFalse(range.contains(-1))
        assertTrue(range.contains(0))
        assertFalse(range.contains(1))
    }

    @Test
    fun toArray() {
        val range = IntRangeCollection(10, 19)
        assertContentEquals((10..19).toList().toIntArray(), range.toArray())
    }

    @Test
    fun smallSequence() {
        val range = IntRangeCollection(2, 3)
        assertEquals(2, range.asSequence().count())
    }

    @Test
    fun randomOnSingleton() {
        val range = IntRangeCollection(5, 5)
        assertEquals(5, range.random(Random))
    }

    @Test
    fun iteratorReentrant() {
        val s = IntRangeCollection(1, 10)
        assertTrue(s.iterator().hasNext())
        assertEquals(10, s.iterator().asSequence().toSet().size)
        assertEquals(10, s.iterator().asSequence().toSet().size)
    }

    @Test
    fun copySame() {
        val s1 = IntRangeCollection(100, 200)
        val s2 = s1.copy()
        assertEquals(s1.size, s2.size)
        for (i in s1) {
            assertTrue(i in s2)
        }
    }

    @Test
    fun permutation() {
        val s1 = IntRangeCollection(10, 20)
        val s2 = s1.permutation(Random).asSequence().toSet()
        assertContentEquals(s1.toArray().also { it.sort() }, s2.toIntArray().also { it.sort() })
    }
}