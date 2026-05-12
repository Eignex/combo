package combo.util

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class IntHashMapTest {

    @Test
    fun createEmpty() {
        val map = IntHashMap()
        assertEquals(0, map.size)
        assertTrue { map.isEmpty() }
    }

    @Test
    fun add() {
        val map = IntHashMap(nullKey = -1)
        entry(4, 4)
        for (i in 0 until 1000) {
            assertEquals(i, map.size)
            map.add(entry(i, i))
        }
    }

    @Test
    fun addDuplicate() {
        val map = IntHashMap(nullKey = -1)
        for (i in 0 until 1000) {
            assertEquals(i, map.size)
            assertEquals(0, map.add(entry(i, 1)))
            assertEquals(1, map.add(entry(i, 2)))
        }
    }

    @Test
    fun containsNotEmpty() {
        val map = IntHashMap()
        assertFalse(map.contains(map.nullKey))
        assertFalse(map.contains(10))
    }

    @Test
    fun containsAfterAdd() {
        val map = IntHashMap()
        map.add(entry(2, 10))
        assertTrue(map.contains(2))
    }

    @Test
    fun createSmallTable() {
        val map = IntHashMap(0)
        for (i in 1..100)
            map[i] = i
        assertEquals(100, map.size)
    }

    @Test
    fun removeMissing() {
        val map = IntHashMap()
        assertEquals(0, map.remove(1))
        assertEquals(0, map.remove(0))
        assertEquals(0, map.remove(-1))
    }

    @Test
    fun removeAndAddAgain() {
        val map = IntHashMap()
        map[2] = 10
        map[8] = 1
        assertEquals(0, map.remove(3))
        assertEquals(2, map.size)
        assertEquals(10, map.remove(2))
        assertEquals(1, map.size)
        assertEquals(0, map.add(entry(2, 2)))
        assertEquals(2, map.size)
    }

    @Test
    fun clear() {
        val map = IntHashMap()
        for (i in 4..10)
            map[i] = i
        map.remove(5)
        map.clear()
        assertEquals(0, map.size)
        map.add(entry(4, 1))
        assertEquals(1, map.size)
    }

    @Test
    fun emptySequence() {
        val map = IntHashMap()
        assertEquals(0, map.asSequence().count())
    }

    @Test
    fun smallSequence() {
        val map = IntHashMap()
        map.add(entry(8, 1))
        map.add(entry(1, 0))
        assertEquals(2, map.asSequence().count())
    }

    @Test
    fun multipleRehash() {
        val map = IntHashMap(2)
        (1..100).forEach { map.add(entry(it, it)) }
        map.clear()
        (1100..1120).forEach { map.add(entry(it, it)) }
        map.clear()
        (200..300).forEach { map.add(entry(it, it)) }
        assertEquals(101, map.size)
        assertFalse(map.contains(1))
        assertTrue(map.contains(200))
    }

    @Test
    fun largeRandomTest() {
        val r = Random
        val all = ArrayList<Int>()
        val map = IntHashMap(nullKey = -1)
        val test = HashMap<Int, Int>()
        for (i in 1..1_000) {
            val n = r.nextInt(1, 10_000)
            val v = r.nextInt(1, Int.MAX_VALUE)
            all.add(n)
            assertEquals(test.put(n, v) ?: 0, map.set(n, v))
            assertEquals(test.put(n, v) ?: 0, map.set(n, v))
            if (r.nextFloat() < 0.1f) {
                val remove = all[r.nextInt(all.size)]
                assertEquals(test.remove(remove) ?: 0, map.remove(remove))
                assertEquals(test.remove(remove) ?: 0, map.remove(remove))
            }
            if (r.nextFloat() < 0.01f) {
                map.clear()
                test.clear()
            }
        }
        for (i in all)
            assertEquals(test.remove(i) ?: 0, map.remove(i))
    }

    @Test
    fun iteratorReentrant() {
        val s = IntHashMap()
        for (i in 1..10) s.add(entry(i, Random.nextInt()))
        assertTrue(s.iterator().hasNext())
        assertEquals(10, s.iterator().asSequence().toSet().size)
        assertEquals(10, s.iterator().asSequence().toSet().size)
    }

    @Test
    fun copySame() {
        val s1 = IntHashMap(4, -1)
        for (i in 1..10) s1.add(entry(Random.nextInt(1, Int.MAX_VALUE), Random.nextInt()))
        val s2 = s1.copy()
        assertEquals(s1.size, s2.size)
        for (entry in s1.entryIterator()) {
            assertTrue(s1.contains(entry.key()))
            assertEquals(s1[entry.key()], s2[entry.key()])
        }
    }
}