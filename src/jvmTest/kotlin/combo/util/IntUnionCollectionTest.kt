package combo.util

import combo.test.assertContentEquals
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class IntUnionCollectionTest {

    @Test
    fun createEmpty() {
        val set = IntUnionCollection(IntHashSet(), IntArrayList())
        assertEquals(0, set.size)
        assertFalse(set.contains(-1))
        assertFalse(set.contains(0))
        assertFalse(set.contains(1))
        assertTrue(set.isEmpty())
        assertTrue(set.toArray().isEmpty())
    }

    @Test
    fun emptyAndOne() {
        val set = IntUnionCollection(IntRangeCollection(0, 0), IntArrayList())
        assertEquals(1, set.size)
        assertFalse(set.isEmpty())
        assertTrue(set.contains(0))
        assertFalse(set.contains(-1))
    }

    @Test
    fun oneAndOne() {
        val set = IntUnionCollection(IntArrayList(intArrayOf(1)), IntArrayList(intArrayOf(1)))
        assertEquals(2, set.size)
        assertTrue(set.contains(1))
    }

    @Test
    fun toArray() {
        val set = IntUnionCollection(IntRangeCollection(10, 15), IntRangeCollection(16, 19))
        assertContentEquals((10..19).toList().toIntArray(), set.toArray())
    }

    @Test
    fun randomOnSingleton() {
        val set = IntRangeCollection(5, 5)
        assertEquals(5, set.random(Random))
    }

    @Test
    fun random() {
        val col = IntUnionCollection(IntArrayList(intArrayOf(0, 1, 2)), IntArrayList(intArrayOf(3)))
        val rng = Random(0)
        val set = generateSequence { col.random(rng) }.take(100).toSet()
        assertEquals(4, set.size)
    }

    @Test
    fun copySame() {
        val s1 = IntUnionCollection(IntArrayList(IntArray(10) { it }), IntArrayList(IntArray(10) { -it }))
        val s2 = s1.copy()
        assertEquals(s1.size, s2.size)
        for (i in s1) {
            assertTrue(i in s2)
        }
    }

    @Test
    fun permutation() {
        val s1 = IntUnionCollection(IntArrayList(IntArray(10) { it }), IntArrayList(IntArray(10) { -it }))
        val s2 = s1.permutation(Random).asSequence().toList()
        assertContentEquals(s1.toArray().also { it.sort() }, s2.toIntArray().also { it.sort() })
    }

    @Test
    fun emptyPermutation() {
        assertEquals(0, IntUnionCollection(IntArrayList(0), IntArrayList(0)).permutation(Random).asSequence().count())
    }
}