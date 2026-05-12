package combo.expressions

import combo.sat.BitArray
import combo.sat.getBits
import combo.test.assertContentEquals
import combo.util.*
import kotlin.random.Random
import kotlin.test.*

class ConjunctionTest : ConstraintTest() {

    @Test
    fun satisfies() {
        val instance = BitArray(4, IntArray(1) { 0b0110 })
        assertFalse(Conjunction(IntArrayList(intArrayOf(1))).satisfies(instance))
        assertTrue(Conjunction(IntArrayList(intArrayOf(2))).satisfies(instance))
        assertFalse(
            Conjunction(IntArrayList(intArrayOf(1, 3))).satisfies(
                instance
            )
        )
        assertFalse(
            Conjunction(IntArrayList(intArrayOf(1, -3))).satisfies(
                instance
            )
        )
        assertFalse(
            Conjunction(IntArrayList(intArrayOf(1, -4))).satisfies(
                instance
            )
        )
        assertTrue(
            Conjunction(IntArrayList(intArrayOf(2, -4))).satisfies(
                instance
            )
        )
    }

    @Test
    fun violations() {
        val instance = BitArray(4, IntArray(1) { 0b0110 })
        assertEquals(
            1,
            Conjunction(IntArrayList(intArrayOf(1))).violations(instance)
        )
        assertEquals(
            0,
            Conjunction(IntArrayList(intArrayOf(2))).violations(instance)
        )
        assertEquals(
            1,
            Conjunction(IntArrayList(intArrayOf(1, 3))).violations(instance)
        )
        assertEquals(
            2,
            Conjunction(IntArrayList(intArrayOf(1, -3))).violations(instance)
        )
        assertEquals(
            1,
            Conjunction(IntArrayList(intArrayOf(1, -4))).violations(instance)
        )
        assertEquals(
            0,
            Conjunction(IntArrayList(intArrayOf(2, -4))).violations(instance)
        )
    }

    @Test
    fun updateCache() {
        val c = Conjunction(IntArrayList(intArrayOf(1, -4)))
        for (k in 0 until 16) {
            val instance = BitArray(4, IntArray(1) { k })
            randomCacheUpdates(instance, c)
        }
    }

    @Test
    fun unitPropagationNone() {
        val a = Conjunction(IntArrayList(intArrayOf(2, -4)))
        val b = a.unitPropagation(-5)
        val c = a.unitPropagation(3)
        assertContentEquals(
            a.literals.toArray().apply { sort() },
            b.literals.toArray().apply { sort() })
        assertContentEquals(
            a.literals.toArray().apply { sort() },
            c.literals.toArray().apply { sort() })
    }

    @Test
    fun unitPropagationComplete() {
        val a = Conjunction(IntArrayList(intArrayOf(2, -4)))
        assertEquals(Tautology, a.unitPropagation(2).unitPropagation(-4))
    }

    @Test
    fun unitPropagationFail() {
        val a = Conjunction(IntArrayList(intArrayOf(1, 2)))
        assertEquals(Empty, a.unitPropagation(-2))
        assertEquals(Empty, a.unitPropagation(-1))
    }

    @Test
    fun randomExhaustivePropagations() {
        randomExhaustivePropagations(Conjunction(collectionOf(-2)))
        randomExhaustivePropagations(Conjunction(collectionOf(1, -2, 5)))
        randomExhaustivePropagations(Conjunction(collectionOf(1, 2, 3, 4)))
        randomExhaustivePropagations(Conjunction(collectionOf(2, 3, 4, 5)))
        randomExhaustivePropagations(Conjunction(collectionOf(-1, -2, -3, -4)))
        randomExhaustivePropagations(Conjunction(collectionOf(2, 3, -4, -5)))
    }

    @Test
    fun coerce() {
        BitArray(100).also {
            val c = Conjunction(collectionOf(1, 7, 5))
            c.coerce(it, Random)
            assertTrue(c.satisfies(it))
            assertTrue(it.isSet(0) && it.isSet(6) && it.isSet(4))
            assertEquals(3, it.iterator().asSequence().count())
        }
        BitArray(100).also {
            val c = Conjunction(collectionOf(-70, -78))
            c.coerce(it, Random)
            assertTrue(c.satisfies(it))
            assertEquals(0, it.iterator().asSequence().count())
            it[69] = true
            it[77] = true
            assertEquals(2, it.iterator().asSequence().count())
            c.coerce(it, Random)
            assertTrue(c.satisfies(it))
            assertEquals(0, it.iterator().asSequence().count())
        }
    }

    @Test
    fun coerceIntRange() {
        BitArray(100).also {
            val c = Conjunction(IntRangeCollection(39, 56))
            c.coerce(it, Random)
            assertTrue(c.satisfies(it))
            assertEquals(0x3FFFF, it.getBits(38, 18))
            assertEquals(18, it.iterator().asSequence().count())
        }

        BitArray(100).also {
            val c = Conjunction(IntRangeCollection(1, 100))
            c.coerce(it, Random)
            assertTrue(c.satisfies(it))
            assertEquals(-1, it.getBits(0, 32))
            assertEquals(-1, it.getBits(32, 32))
            assertEquals(-1, it.getBits(64, 32))
            assertEquals(15, it.getBits(96, 4))
            assertEquals(100, it.iterator().asSequence().count())
        }

        BitArray(199).also {
            val c = Conjunction(IntRangeCollection(-10, -1))
            c.coerce(it, Random)
            assertTrue(c.satisfies(it))
            assertEquals(0, it.getBits(0, 10))
            for (i in 0 until 10)
                it[i] = true
            assertEquals(10, Int.bitCount(it.getBits(0, 10)))
            Conjunction(IntRangeCollection(-10, -1)).coerce(it, Random)
            assertEquals(0, Int.bitCount(it.getBits(0, 10)))
        }
    }

    @Test
    fun randomCoerce() {
        randomCoerce(Conjunction(collectionOf(1, 4, 5)))
        randomCoerce(Conjunction(collectionOf(1, -4, 5)))
        randomCoerce(Conjunction(collectionOf(1, 5)))
        randomCoerce(Conjunction(collectionOf(1, -5)))
        randomCoerce(Conjunction(collectionOf(-3)))
    }
}

class DisjunctionTest : ConstraintTest() {

    @Test
    fun satisfies() {
        val instance = BitArray(4, IntArray(1) { 0b0110 })
        assertFalse(
            Disjunction(IntArrayList(intArrayOf(-2, -3))).satisfies(
                instance
            )
        )
        assertTrue(
            Disjunction(IntArrayList(intArrayOf(-2, -1))).satisfies(
                instance
            )
        )
        assertTrue(
            Disjunction(IntArrayList(intArrayOf(1, 3))).satisfies(
                instance
            )
        )
        assertFalse(
            Disjunction(IntArrayList(intArrayOf(1, -3))).satisfies(
                instance
            )
        )
        assertTrue(
            Disjunction(IntArrayList(intArrayOf(1, -4))).satisfies(
                instance
            )
        )
    }

    @Test
    fun violations() {
        val instance = BitArray(4, IntArray(1) { 0b0110 })
        assertEquals(
            1,
            Disjunction(IntArrayList(intArrayOf(-2, -3))).violations(instance)
        )
        assertEquals(
            0,
            Disjunction(IntArrayList(intArrayOf(-2, -1))).violations(instance)
        )
        assertEquals(
            0,
            Disjunction(IntArrayList(intArrayOf(1, 3))).violations(instance)
        )
        assertEquals(
            1,
            Disjunction(IntArrayList(intArrayOf(1, -3))).violations(instance)
        )
        assertEquals(
            0,
            Disjunction(IntArrayList(intArrayOf(1, -4))).violations(instance)
        )
    }

    @Test
    fun updateCache() {
        val c = Disjunction(IntArrayList(intArrayOf(1, -4)))
        for (k in 0 until 16) {
            val instance = BitArray(4, IntArray(1) { k })
            randomCacheUpdates(instance, c)
        }
    }

    @Test
    fun unitPropagation() {
        val a = Disjunction(IntArrayList(intArrayOf(1, 2, 3)))
        val a1 = a.unitPropagation(2)
        assertEquals(a1, Tautology)
        val a2 = a.unitPropagation(-2)
        assertEquals(2, a2.literals.size)
        assertTrue(1 in a2.literals)
        assertFalse(2 in a2.literals)
        assertTrue(3 in a2.literals)

        val b = Disjunction(IntArrayList(intArrayOf(-1, 2, 3)))
        val d1 = b.unitPropagation(-1)
        assertEquals(d1, Tautology)
        val d2 = b.unitPropagation(1)
        assertEquals(2, d2.literals.size)
        assertFalse(-1 in d2.literals)
        assertTrue(2 in d2.literals)
        assertTrue(3 in d2.literals)
    }

    @Test
    fun unitPropagationNone() {
        val a = Disjunction(IntArrayList(intArrayOf(-1, -4)))
        val b = a.unitPropagation(-5)
        assertContentEquals(
            a.literals.toArray().apply { sort() },
            b.literals.toArray().apply { sort() })
    }

    @Test
    fun unitPropagationFail() {
        val a = Disjunction(IntArrayList(intArrayOf(2, -3)))
        assertEquals(Empty, a.unitPropagation(-2).unitPropagation(3))
    }

    @Test
    fun randomExhaustivePropagations() {
        randomExhaustivePropagations(Disjunction(collectionOf(-2)))
        randomExhaustivePropagations(Disjunction(collectionOf(1, -2, 5)))
        randomExhaustivePropagations(Disjunction(collectionOf(1, 2, 3, 4)))
        randomExhaustivePropagations(Disjunction(collectionOf(2, 3, 4, 5)))
        randomExhaustivePropagations(Disjunction(collectionOf(-1, -2, -3, -4)))
        randomExhaustivePropagations(Disjunction(collectionOf(-1, 3, -4, -5)))
    }

    @Test
    fun coerce() {
        BitArray(100).also {
            val c = Disjunction(collectionOf(1, 7, 5))
            c.coerce(it, Random)
            assertTrue(c.satisfies(it))
            assertTrue(it.isSet(0) || it.isSet(6) || it.isSet(4))
            assertEquals(1, it.iterator().asSequence().count())
        }
        BitArray(100).also {
            val c = Disjunction(collectionOf(-70, -78))
            c.coerce(it, Random)
            assertTrue(c.satisfies(it))
            assertEquals(0, it.iterator().asSequence().count())
            it[69] = true
            it[77] = true
            assertEquals(2, it.iterator().asSequence().count())
            c.coerce(it, Random)
            assertTrue(c.satisfies(it))
            assertEquals(1, it.iterator().asSequence().count())
        }
    }

    @Test
    fun coerceIntRange() {
        BitArray(100).also {
            val c = Disjunction(IntRangeCollection(39, 56))
            c.coerce(it, Random)
            assertTrue(c.satisfies(it))
            assertTrue(it.getBits(38, 18) != 0)
            assertEquals(1, it.iterator().asSequence().count())
        }

        BitArray(100).also {
            val c = Disjunction(IntRangeCollection(1, 100))
            c.coerce(it, Random)
            assertTrue(c.satisfies(it))
            assertTrue(
                it.getBits(0, 32) != 0 || it.getBits(
                    32,
                    32
                ) != 0 || it.getBits(64, 32) != 0 || it.getBits(96, 4) != 0
            )
            assertEquals(1, it.iterator().asSequence().count())
        }

        BitArray(199).also {
            val c = Disjunction(IntRangeCollection(-10, -1))
            c.coerce(it, Random)
            assertTrue(c.satisfies(it))
            assertEquals(0, it.getBits(0, 10))
            for (i in 0 until 10)
                it[i] = true
            assertEquals(10, Int.bitCount(it.getBits(0, 10)))
            Disjunction(IntRangeCollection(-10, -1)).coerce(it, Random)
            assertEquals(9, Int.bitCount(it.getBits(0, 10)))
        }
    }

    @Test
    fun randomCoerce() {
        randomCoerce(Disjunction(collectionOf(1, 4, 5)))
        randomCoerce(Disjunction(collectionOf(1, -4, 5)))
        randomCoerce(Disjunction(collectionOf(1, 5)))
        randomCoerce(Disjunction(collectionOf(1, -5)))
        randomCoerce(Disjunction(collectionOf(-3)))
    }
}
