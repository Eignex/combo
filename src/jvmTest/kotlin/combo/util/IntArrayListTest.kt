package combo.util

import combo.test.assertContentEquals
import kotlin.random.Random
import kotlin.test.*

class IntArrayListTest {

    @Test
    fun createEmpty() {
        val list = IntArrayList()
        assertEquals(0, list.size)
        assertTrue { list.isEmpty() }
    }

    @Test
    fun add() {
        val list = IntArrayList()
        for (i in 0 until 1000) {
            assertEquals(i, list.size)
            list.add(i)
        }
    }

    @Test
    fun containsNotEmpty() {
        val list = IntArrayList()
        assertFalse(list.contains(0))
    }

    @Test
    fun containsAfterAdd() {
        val list = IntArrayList()
        list.add(2)
        assertTrue(list.contains(2))
    }

    @Test
    fun addAllIntArray() {
        val list = IntArrayList()
        list.addAll(intArrayOf(2, 4))
        assertTrue(list.contains(2))
        assertTrue(list.contains(4))
    }

    @Test
    fun addAllIterable() {
        val list = IntArrayList()
        assertTrue(list.addAll((2..4).asIterable()))
        assertTrue(list.contains(2))
        assertTrue(list.contains(3))
        assertTrue(list.contains(4))
    }

    @Test
    fun removeMissingFromSet() {
        val list = IntArrayList()
        assertFalse(list.remove(1))
        assertFalse(list.remove(-1))
    }

    @Test
    fun removeFromSetAndAddAgain() {
        val list = IntArrayList()
        list.add(2)
        list.add(8)
        assertFalse(list.remove(3))
        assertEquals(2, list.size)
        assertTrue(list.remove(2))
        assertEquals(1, list.size)
        assertTrue(list.add(2))
        assertEquals(2, list.size)
    }

    @Test
    fun toArrayOnEmpty() {
        val list = IntArrayList()
        assertTrue { list.toArray().isEmpty() }
    }

    @Test
    fun toArrayOnRemoved() {
        val list = IntArrayList()
        list.add(0)
        assertEquals(1, list.toArray().size)
        list.remove(0)
        assertEquals(0, list.toArray().size)
    }

    @Test
    fun clear() {
        val list = IntArrayList()
        for (i in 4..10)
            list.add(i)
        list.remove(5)
        list.clear()
        assertEquals(0, list.size)
        list.add(4)
        assertEquals(1, list.size)
    }

    @Test
    fun emptySequence() {
        val list = IntArrayList()
        assertEquals(0, list.asSequence().count())
    }

    @Test
    fun smallSequence() {
        val list = IntArrayList()
        list.add(8)
        list.add(1)
        assertEquals(2, list.asSequence().count())
    }

    @Test
    fun randomOnEmpty() {
        assertFailsWith(IllegalArgumentException::class) {
            val list = IntArrayList()
            list.random(Random)
        }
    }

    @Test
    fun randomOnSingleton() {
        val list = IntArrayList()
        list.add(12300)
        assertEquals(12300, list.random(Random))
    }

    @Test
    fun multipleRehash() {
        val list = IntArrayList(2)
        list.addAll((1..100).asSequence().asIterable())
        list.clear()
        list.addAll((1100..1120).asSequence().asIterable())
        list.clear()
        list.addAll((200..300).asSequence().asIterable())
        assertEquals(101, list.size)
        assertFalse(list.contains(1))
        assertTrue(list.contains(200))
    }

    @Test
    fun largeRandomTest() {
        val r = Random(0)
        val all = ArrayList<Int>()
        val list = IntArrayList()
        val test = HashSet<Int>()
        for (i in 1..1_000) {
            val n = r.nextInt(Int.MAX_VALUE)
            all.add(n)
            assertEquals(test.add(n), list.add(n))
            if (r.nextBoolean()) {
                val remove = all[r.nextInt(all.size)]
                assertEquals(test.remove(remove), list.remove(remove))
                assertEquals(test.remove(remove), list.remove(remove))
            }
        }
        for (i in all)
            assertEquals(test.remove(i), list.remove(i))
    }

    @Test
    fun iterator() {
        val s = IntArrayList()
        s.addAll(generateSequence { Random.nextInt(0, Int.MAX_VALUE / 2) }.take(10).asIterable())
        assertTrue(s.iterator().hasNext())
        assertEquals(10, s.iterator().asSequence().toSet().size)
        assertEquals(10, s.iterator().asSequence().toSet().size)
    }

    @Test
    fun copySame() {
        val s1 = IntArrayList()
        s1.addAll(generateSequence { Random.nextInt(0, Int.MAX_VALUE / 2) }.take(10).asIterable())
        val s2 = s1.copy()
        assertEquals(s1.size, s2.size)
        for (i in s1) {
            assertTrue(i in s2)
        }
    }

    @Test
    fun permutation() {
        val s1 = IntArrayList()
        s1.addAll(10..20)
        val s2 = s1.permutation(Random).asSequence().toSet()
        assertContentEquals(s1.toArray().also { it.sort() }, s2.toIntArray().also { it.sort() })
    }

    @Test
    fun emptyPermutation() {
        assertEquals(0, IntArrayList().permutation(Random).asSequence().count())
    }
}