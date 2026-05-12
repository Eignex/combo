package combo.util

import combo.test.assertContentEquals
import kotlin.random.Random
import kotlin.test.*

class IntHashSetTest {

    @Test
    fun createEmpty() {
        val set = IntHashSet()
        assertEquals(0, set.size)
        assertTrue { set.isEmpty() }
    }

    @Test
    fun add() {
        val set = IntHashSet(nullValue = -1)
        for (i in 0 until 1000) {
            assertEquals(i, set.size)
            set.add(i)
        }
    }

    @Test
    fun addDuplicate() {
        val set = IntHashSet(nullValue = -1)
        for (i in 0 until 1000) {
            assertEquals(i, set.size)
            assertTrue(set.add(i))
            assertFalse(set.add(i))
        }
    }

    @Test
    fun containsNotEmpty() {
        val set = IntHashSet()
        assertFalse(set.contains(0))
    }

    @Test
    fun containsAfterAdd() {
        val set = IntHashSet()
        set.add(2)
        assertTrue(set.contains(2))
    }

    @Test
    fun addAllIntArray() {
        val set = IntHashSet()
        set.addAll(intArrayOf(2, 4))
        assertTrue(set.contains(2))
        assertTrue(set.contains(4))
    }

    @Test
    fun addAllIterable() {
        val set = IntHashSet()
        assertTrue(set.addAll((2..4).asIterable()))
        assertFalse(set.addAll((2..4).asIterable()))
        assertTrue(set.contains(2))
        assertTrue(set.contains(3))
        assertTrue(set.contains(4))
    }

    @Test
    fun removeMissingFromSet() {
        val set = IntHashSet()
        assertFalse(set.remove(1))
        assertFalse(set.remove(-1))
    }

    @Test
    fun removeFromSetAndAddAgain() {
        val set = IntHashSet()
        set.add(2)
        set.add(8)
        assertFalse(set.remove(3))
        assertEquals(2, set.size)
        assertTrue(set.remove(2))
        assertEquals(1, set.size)
        assertTrue(set.add(2))
        assertEquals(2, set.size)
    }

    @Test
    fun createSmallTable() {
        val set = IntHashSet(0)
        for (i in 1..100)
            set.add(i)
        assertEquals(100, set.size)
    }

    @Test
    fun toArrayOnEmpty() {
        val set = IntHashSet()
        assertTrue { set.toArray().isEmpty() }
    }

    @Test
    fun toArrayOnRemoved() {
        val set = IntHashSet(nullValue = -1)
        set.add(0)
        assertEquals(1, set.toArray().size)
        set.remove(0)
        assertEquals(0, set.toArray().size)
    }

    @Test
    fun clear() {
        val set = IntHashSet()
        for (i in 4..10)
            set.add(i)
        set.remove(5)
        set.clear()
        assertEquals(0, set.size)
        set.add(4)
        assertEquals(1, set.size)
    }

    @Test
    fun emptySequence() {
        val set = IntHashSet()
        assertEquals(0, set.asSequence().count())
    }

    @Test
    fun smallSequence() {
        val set = IntHashSet()
        set.add(8)
        set.add(1)
        assertEquals(2, set.asSequence().count())
    }

    @Test
    fun randomOnEmpty() {
        assertFailsWith(NoSuchElementException::class) {
            val set = IntHashSet()
            set.random(Random)
        }
    }

    @Test
    fun randomOnSingleton() {
        val set = IntHashSet()
        set.add(12300)
        assertEquals(12300, set.random(Random))
    }

    @Test
    fun multipleRehash() {
        val set = IntHashSet(2)
        set.addAll((1..100).asSequence().asIterable())
        set.clear()
        set.addAll((1100..1120).asSequence().asIterable())
        set.clear()
        set.addAll((200..300).asSequence().asIterable())
        assertEquals(101, set.size)
        assertFalse(set.contains(1))
        assertTrue(set.contains(200))
    }

    @Test
    fun largeRandomTest() {
        val r = Random
        val all = ArrayList<Int>()
        val set = IntHashSet(nullValue = -1)
        val test = HashSet<Int>()
        for (i in 1..1_000) {
            val n = r.nextInt(10_000)
            all.add(n)
            assertEquals(test.add(n), set.add(n))
            assertEquals(test.add(n), set.add(n))
            if (r.nextFloat() < 0.1f) {
                val remove = all[r.nextInt(all.size)]
                assertEquals(test.remove(remove), set.remove(remove))
                assertEquals(test.remove(remove), set.remove(remove))
            }
            if (r.nextFloat() < 0.01f) {
                set.clear()
                test.clear()
            }
        }
        for (i in all)
            assertEquals(test.remove(i), set.remove(i))
    }

    @Test
    fun iteratorReentrant() {
        val s = IntHashSet()
        s.addAll(IntArray(10) { it + 1 })
        assertTrue(s.iterator().hasNext())
        assertEquals(10, s.iterator().asSequence().toSet().size)
        assertEquals(10, s.iterator().asSequence().toSet().size)
    }

    @Test
    fun copySame() {
        val s1 = IntHashSet()
        s1.addAll(generateSequence { Random.nextInt(0, Int.MAX_VALUE / 2) }.take(10).asIterable())
        val s2 = s1.copy()
        assertEquals(s1.size, s2.size)
        for (i in s1) {
            assertTrue(i in s2)
        }
    }

    @Test
    fun permutation() {
        val s1 = IntHashSet()
        s1.addAll(10..20)
        val s2 = s1.permutation(Random).asSequence().toSet()
        assertContentEquals(s1.toArray().also { it.sort() }, s2.toIntArray().also { it.sort() })
    }

    @Test
    fun emptyPermutation() {
        assertEquals(0, IntHashSet().permutation(Random).asSequence().count())
    }
}