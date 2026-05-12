package combo.expressions

import combo.expressions.Relation.*
import combo.sat.*
import combo.test.assertContentEquals
import combo.util.*
import kotlin.math.*
import kotlin.random.Random
import kotlin.test.*

class LinearTest : ConstraintTest() {

    @Test
    fun satisfiesBlank() {
        val instance = SparseBitArray(4)
        val linear = Linear(IntHashMap().apply {
            this[1] = 0
            this[-2] = 1
            this[3] = 2
        }, intArrayOf(2, 4, -1), 4, EQ)
        assertTrue(linear.satisfies(instance))
        assertEquals(0, linear.violations(instance))
    }

    @Test
    fun satisfies1() {
        val weights = intArrayOf(50, -200, -50)
        val e = Linear(IntHashMap().apply {
            this[1] = 0
            this[2] = 1
            this[3] = 2
        }, weights, -50, LE)
        // 50x1 - 200x2 - 50x3 <= -50
        assertFalse(e.satisfies(BitArray(3, IntArray(1) { 0b000 })))
        assertFalse(e.satisfies(BitArray(3, IntArray(1) { 0b001 })))
        assertTrue(e.satisfies(BitArray(3, IntArray(1) { 0b010 })))
        assertTrue(e.satisfies(BitArray(3, IntArray(1) { 0b011 })))
        assertTrue(e.satisfies(BitArray(3, IntArray(1) { 0b100 })))
        assertFalse(e.satisfies(BitArray(3, IntArray(1) { 0b101 })))
        assertTrue(e.satisfies(BitArray(3, IntArray(1) { 0b110 })))
        assertTrue(e.satisfies(BitArray(3, IntArray(1) { 0b111 })))
    }

    @Test
    fun satisfies2() {
        val e = Linear(IntHashMap().apply {
            this[1] = 0
            this[2] = 1
            this[-3] = 2
        }, intArrayOf(10, -10, -10), 10, GE)
        // 10x1 - 10x2 - 10!x3 >= 10
        assertFalse(e.satisfies(BitArray(3, IntArray(1) { 0b000 })))
        assertFalse(e.satisfies(BitArray(3, IntArray(1) { 0b001 })))
        assertFalse(e.satisfies(BitArray(3, IntArray(1) { 0b010 })))
        assertFalse(e.satisfies(BitArray(3, IntArray(1) { 0b011 })))
        assertFalse(e.satisfies(BitArray(3, IntArray(1) { 0b100 })))
        assertTrue(e.satisfies(BitArray(3, IntArray(1) { 0b101 })))
        assertFalse(e.satisfies(BitArray(3, IntArray(1) { 0b110 })))
        assertFalse(e.satisfies(BitArray(3, IntArray(1) { 0b111 })))
    }

    @Test
    fun violationsAsCardinality() {
        val literals1 = collectionOf(-3, 2, -1, 4)
        val weights = IntArray(4) { 1 }
        val literals2 = IntHashMap().apply {
            this[-3] = 0; this[2] = 1; this[-1] = 2; this[4] = 3
        }
        for (degree in 0..5) {
            for (rel in entries) {
                val c1 = Cardinality(literals1, degree, rel)
                val c2 = Linear(literals2, weights, degree, rel)
                for (instanceI in 0 until 16) {
                    val instance = BitArray(4, intArrayOf(instanceI))
                    assertEquals(
                        c1.violations(instance),
                        c2.violations(instance)
                    )
                }
            }
        }
    }

    @Test
    fun updateCache() {
        for (rel in entries) {
            val c = Linear(IntHashMap().apply {
                this[1] = 0
                this[-2] = 1
                this[3] = 2
            }, intArrayOf(-2, -1, 1), 1, rel)
            for (k in 0 until 16) {
                val instance = BitArray(4, IntArray(1) { k })
                randomCacheUpdates(instance, c)
            }
        }
    }

    @Test
    fun unitPropagationNone() {
        val a = Linear(IntHashMap().apply {
            this[1] = 0
            this[5] = 1
            this[6] = 2
        }, intArrayOf(2, 3, -1), 1, LE)
        assertContentEquals(
            a.literals.toArray().apply { sort() },
            a.unitPropagation(2).literals.toArray().apply { sort() })
        assertContentEquals(
            a.literals.toArray().apply { sort() },
            a.unitPropagation(4).literals.toArray().apply { sort() })
        assertContentEquals(
            a.literals.toArray().apply { sort() },
            a.unitPropagation(8).literals.toArray().apply { sort() })
    }

    @Test
    fun randomExhaustivePropagations() {
        val lits = IntHashMap()
        lits[1] = 0
        lits[2] = 1
        lits[3] = 2
        lits[-4] = 3
        lits[5] = 4

        val weights = arrayOf(
            IntArray(5) { it - 1 },
            IntArray(5) { 1 },
            IntArray(5) { -1 },
            IntArray(5) { Random(0).nextInt(-10, 10) }
        )
        for (rel in entries) {
            for (w in weights) {
                for (deg in -5..5) {
                    randomExhaustivePropagations(Linear(lits, w, deg, rel))
                }
            }
        }
    }

    @Test
    fun coerce() {
        BitArray(100).also {
            val c = Cardinality(collectionOf(1, 7, 5), 1, GT)
            c.coerce(it, Random)
            assertTrue(c.satisfies(it))
            assertTrue(it.isSet(0) || it.isSet(6) || it.isSet(4))
            assertEquals(2, it.iterator().asSequence().count())
        }
        BitArray(100).also {
            val c = Cardinality(collectionOf(-70, -78), 1, EQ)
            c.coerce(it, Random)
            assertTrue(c.satisfies(it))
            assertEquals(1, it.iterator().asSequence().count())
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
            val c = Cardinality(IntRangeCollection(39, 56), 0, NE)
            c.coerce(it, Random)
            assertTrue(c.satisfies(it))
            assertTrue(it.getBits(38, 18) != 0)
            assertEquals(1, it.iterator().asSequence().count())
        }

        BitArray(100).also {
            for (i in it.indices) it[i] = true
            val c = Cardinality(IntRangeCollection(1, 100), 10, LT)
            c.coerce(it, Random)
            assertTrue(c.satisfies(it))
            assertEquals(9, it.iterator().asSequence().count())
        }

        BitArray(199).also {
            val c = Cardinality(IntRangeCollection(-10, -1), 2, GE)
            for (i in 0 until 10) it[i] = true
            c.coerce(it, Random)
            assertTrue(c.satisfies(it))
            assertEquals(8, Int.bitCount(it.getBits(0, 10)))
        }
    }

    @Test
    fun randomCoerce() {
        randomCoerce(Cardinality(collectionOf(1, 2, 3, 4), 2, EQ))
        randomCoerce(Cardinality(collectionOf(2, 3), 1, EQ))
        randomCoerce(Cardinality(collectionOf(1, 2, 3, 4), 2, NE))
        randomCoerce(Cardinality(collectionOf(2, 3), 1, NE))
        randomCoerce(Cardinality(collectionOf(1, 2, 3, 4), 2, LE))
        randomCoerce(Cardinality(collectionOf(2, 3), 1, LE))
        randomCoerce(Cardinality(collectionOf(1, 2, 3, 4), 2, LT))
        randomCoerce(Cardinality(collectionOf(2, 3), 1, LT))
        randomCoerce(Cardinality(collectionOf(1, 2, 3, 4), 2, GE))
        randomCoerce(Cardinality(collectionOf(2, 3), 1, GE))
        randomCoerce(Cardinality(collectionOf(1, 2, 3, 4), 2, GT))
        randomCoerce(Cardinality(collectionOf(2, 3), 1, GT))
    }
}

class CardinalityTest : ConstraintTest() {

    @Test
    fun satisfiesBlank() {
        val instance = SparseBitArray(4)
        assertTrue(
            Cardinality(
                IntArrayList(intArrayOf(1, -2, 3)),
                1,
                LE
            ).satisfies(instance)
        )
        assertEquals(
            0,
            Cardinality(collectionOf(1, -2, 3), 1, LE).violations(instance)
        )
    }

    @Test
    fun satisfies1() {
        val e = Cardinality(collectionOf(1, 2, 3), 1, LE)
        assertTrue(e.satisfies(BitArray(3, IntArray(1) { 0b000 })))
        assertTrue(e.satisfies(BitArray(3, IntArray(1) { 0b001 })))
        assertTrue(e.satisfies(BitArray(3, IntArray(1) { 0b010 })))
        assertFalse(e.satisfies(BitArray(3, IntArray(1) { 0b011 })))
        assertTrue(e.satisfies(BitArray(3, IntArray(1) { 0b100 })))
        assertFalse(e.satisfies(BitArray(3, IntArray(1) { 0b101 })))
        assertFalse(e.satisfies(BitArray(3, IntArray(1) { 0b110 })))
        assertFalse(e.satisfies(BitArray(3, IntArray(1) { 0b111 })))
    }

    @Test
    fun satisfies2() {
        val e = Cardinality(collectionOf(1, 2, -3), 1, LE)
        assertTrue(e.satisfies(BitArray(3, IntArray(1) { 0b000 })))
        assertFalse(e.satisfies(BitArray(3, IntArray(1) { 0b001 })))
        assertFalse(e.satisfies(BitArray(3, IntArray(1) { 0b010 })))
        assertFalse(e.satisfies(BitArray(3, IntArray(1) { 0b011 })))
        assertTrue(e.satisfies(BitArray(3, IntArray(1) { 0b100 })))
        assertTrue(e.satisfies(BitArray(3, IntArray(1) { 0b101 })))
        assertTrue(e.satisfies(BitArray(3, IntArray(1) { 0b110 })))
        assertFalse(e.satisfies(BitArray(3, IntArray(1) { 0b111 })))
    }

    @Test
    fun violations() {
        val instances = arrayOf(
            BitArray(3, IntArray(1) { 0b000 }),
            BitArray(3, IntArray(1) { 0b001 }),
            BitArray(3, IntArray(1) { 0b101 }),
            BitArray(3, IntArray(1) { 0b111 })
        )

        fun testConstraint(
            degree: Int,
            relation: Relation,
            expectedMatches: IntArray
        ) {
            val c = Cardinality(collectionOf(1, 2, 3), degree, relation)
            for (i in 0..3)
                assertEquals(
                    expectedMatches[i],
                    c.violations(instances[i]),
                    "$i"
                )
        }

        testConstraint(0, LE, intArrayOf(0, 1, 2, 3))
        testConstraint(1, LE, intArrayOf(0, 0, 1, 2))
        testConstraint(2, LE, intArrayOf(0, 0, 0, 1))
        testConstraint(3, LE, intArrayOf(0, 0, 0, 0))

        testConstraint(1, LT, intArrayOf(0, 1, 2, 3))
        testConstraint(2, LT, intArrayOf(0, 0, 1, 2))
        testConstraint(3, LT, intArrayOf(0, 0, 0, 1))
        testConstraint(4, LT, intArrayOf(0, 0, 0, 0))

        testConstraint(0, EQ, intArrayOf(0, 1, 2, 3))
        testConstraint(1, EQ, intArrayOf(1, 0, 1, 2))
        testConstraint(2, EQ, intArrayOf(2, 1, 0, 1))
        testConstraint(3, EQ, intArrayOf(3, 2, 1, 0))

        testConstraint(0, NE, intArrayOf(1, 0, 0, 0))
        testConstraint(1, NE, intArrayOf(0, 1, 0, 0))
        testConstraint(2, NE, intArrayOf(0, 0, 1, 0))
        testConstraint(3, NE, intArrayOf(0, 0, 0, 1))

        testConstraint(0, GE, intArrayOf(0, 0, 0, 0))
        testConstraint(1, GE, intArrayOf(1, 0, 0, 0))
        testConstraint(2, GE, intArrayOf(2, 1, 0, 0))
        testConstraint(3, GE, intArrayOf(3, 2, 1, 0))

        testConstraint(0, GT, intArrayOf(1, 0, 0, 0))
        testConstraint(1, GT, intArrayOf(2, 1, 0, 0))
        testConstraint(2, GT, intArrayOf(3, 2, 1, 0))
    }

    @Test
    fun updateCache() {
        val c = Cardinality(IntArrayList(intArrayOf(1, -2, 3)), 1, EQ)
        for (k in 0 until 16) {
            val instance = BitArray(4, IntArray(1) { k })
            randomCacheUpdates(instance, c)
        }
    }

    @Test
    fun unitPropagationNone() {
        val a = Cardinality(IntArrayList(intArrayOf(1, 5, 6)), 1, LE)
        assertContentEquals(
            a.literals.toArray().apply { sort() },
            a.unitPropagation(2).literals.toArray().apply { sort() })
        assertContentEquals(
            a.literals.toArray().apply { sort() },
            a.unitPropagation(4).literals.toArray().apply { sort() })
        assertContentEquals(
            a.literals.toArray().apply { sort() },
            a.unitPropagation(8).literals.toArray().apply { sort() })
    }

    @Test
    fun randomExhaustivePropagations() {
        val lits = collectionOf(1, 2, 3, 4, 5)
        randomExhaustivePropagations(Cardinality(lits, 1, LE))
        randomExhaustivePropagations(Cardinality(lits, 3, LE))
        randomExhaustivePropagations(Cardinality(lits, 1, LT))
        randomExhaustivePropagations(Cardinality(lits, 3, LT))
        randomExhaustivePropagations(Cardinality(lits, 1, GE))
        randomExhaustivePropagations(Cardinality(lits, 3, GE))
        randomExhaustivePropagations(Cardinality(lits, 1, GT))
        randomExhaustivePropagations(Cardinality(lits, 3, GT))
        randomExhaustivePropagations(Cardinality(lits, 1, EQ))
        randomExhaustivePropagations(Cardinality(lits, 3, EQ))
        randomExhaustivePropagations(Cardinality(lits, 1, NE))
        randomExhaustivePropagations(Cardinality(lits, 3, NE))
    }

    @Test
    fun coerce() {
        BitArray(100).also {
            val c = Cardinality(collectionOf(1, 7, 5), 1, GT)
            c.coerce(it, Random)
            assertTrue(c.satisfies(it))
            assertTrue(it.isSet(0) || it.isSet(6) || it.isSet(4))
            assertEquals(2, it.iterator().asSequence().count())
        }
        BitArray(100).also {
            val c = Cardinality(collectionOf(-70, -78), 1, EQ)
            c.coerce(it, Random)
            assertTrue(c.satisfies(it))
            assertEquals(1, it.iterator().asSequence().count())
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
            val c = Cardinality(IntRangeCollection(39, 56), 0, NE)
            c.coerce(it, Random)
            assertTrue(c.satisfies(it))
            assertTrue(it.getBits(38, 18) != 0)
            assertEquals(1, it.iterator().asSequence().count())
        }

        BitArray(100).also {
            for (i in it.indices) it[i] = true
            val c = Cardinality(IntRangeCollection(1, 100), 10, LT)
            c.coerce(it, Random)
            assertTrue(c.satisfies(it))
            assertEquals(9, it.iterator().asSequence().count())
        }

        BitArray(199).also {
            val c = Cardinality(IntRangeCollection(-10, -1), 2, GE)
            for (i in 0 until 10) it[i] = true
            c.coerce(it, Random)
            assertTrue(c.satisfies(it))
            assertEquals(8, Int.bitCount(it.getBits(0, 10)))
        }
    }

    @Test
    fun randomCoerce() {
        randomCoerce(Cardinality(collectionOf(1, 2, 3, 4), 2, EQ))
        randomCoerce(Cardinality(collectionOf(2, 3), 1, EQ))
        randomCoerce(Cardinality(collectionOf(1, 2, 3, 4), 2, NE))
        randomCoerce(Cardinality(collectionOf(2, 3), 1, NE))
        randomCoerce(Cardinality(collectionOf(1, 2, 3, 4), 2, LE))
        randomCoerce(Cardinality(collectionOf(2, 3), 1, LE))
        randomCoerce(Cardinality(collectionOf(1, 2, 3, 4), 2, LT))
        randomCoerce(Cardinality(collectionOf(2, 3), 1, LT))
        randomCoerce(Cardinality(collectionOf(1, 2, 3, 4), 2, GE))
        randomCoerce(Cardinality(collectionOf(2, 3), 1, GE))
        randomCoerce(Cardinality(collectionOf(1, 2, 3, 4), 2, GT))
        randomCoerce(Cardinality(collectionOf(2, 3), 1, GT))
    }
}

class RelationTest {

    @Test
    fun eqRelation() {
        assertEquals(1, EQ.violations(3, 2))
        assertEquals(1, EQ.violations(1, 2))
        assertEquals(0, EQ.violations(2, 2))
        assertEquals(0, EQ.violations(0, 0))
    }

    @Test
    fun neRelation() {
        assertEquals(0, NE.violations(3, 2))
        assertEquals(0, NE.violations(1, 2))
        assertEquals(0, NE.violations(0, 2))
        assertEquals(1, NE.violations(0, 0))
    }

    @Test
    fun leRelation() {
        assertEquals(1, LE.violations(3, 2))
        assertEquals(0, LE.violations(1, 2))
        assertEquals(0, LE.violations(2, 2))
        assertEquals(0, LE.violations(0, 0))
    }

    @Test
    fun ltRelation() {
        assertEquals(2, LT.violations(3, 2))
        assertEquals(0, LT.violations(1, 2))
        assertEquals(1, LT.violations(2, 2))
        assertEquals(1, LT.violations(0, 0))
    }

    @Test
    fun geRelation() {
        assertEquals(0, GE.violations(3, 2))
        assertEquals(1, GE.violations(1, 2))
        assertEquals(2, GE.violations(0, 2))
        assertEquals(0, GE.violations(0, 0))
    }

    @Test
    fun gtRelation() {
        assertEquals(0, GT.violations(3, 2))
        assertEquals(2, GT.violations(1, 2))
        assertEquals(3, GT.violations(0, 2))
        assertEquals(1, GT.violations(0, 0))
    }

    @Test
    fun exhaustiveEmptyConstraint() {
        fun test(weights: IntArray, relation: Relation, degree: Int) {
            val lowerBound = weights.sumOf { min(0, it) }
            val upperBound = weights.sumOf { max(0, it) }
            val n = 2.0.pow(weights.size).roundToInt()
            var allSatisfied = true
            var noneSatisfied = true
            for (i in 0 until n) {
                val instance = BitArray(weights.size, IntArray(1) { i })
                var sum = 0
                for (j in instance)
                    sum += weights[j]
                allSatisfied =
                    allSatisfied && relation.violations(sum, degree) == 0
                noneSatisfied =
                    noneSatisfied && relation.violations(sum, degree) > 0
            }
            assertEquals(
                noneSatisfied,
                relation.isEmpty(lowerBound, upperBound, degree)
            )
            assertEquals(
                allSatisfied,
                relation.isTautology(lowerBound, upperBound, degree)
            )
        }
        for (rel in entries) {
            for (degree in -1..5)
                test(IntArray(4) { 1 }, rel, degree)
        }
        for (rel in entries) {
            for (degree in -4..2)
                test(IntArray(4) { it - 2 }, rel, degree)
        }
    }

    @Test
    fun negation() {
        for (rel in entries)
            for (i in -5..5)
                for (j in -5..5)
                    for (k in -5..5)
                        assertNotEquals(
                            rel.violations(i, j),
                            (!rel).violations(i, j),
                            "$rel $i $j"
                        )
    }

    @Test
    fun noLiteralsIsEmptyOrTautology() {
        for (rel in entries)
            for (k in -5..5)
                assertTrue(rel.isEmpty(0, 0, k) || rel.isTautology(0, 0, k))
    }

    @Test
    fun exhaustiveIsUnit() {
        fun test(weights: IntArray, relation: Relation, degree: Int) {
            val n = 2.0.pow(weights.size).roundToInt()
            val lowerBound = weights.sumOf { min(0, it) }
            val upperBound = weights.sumOf { max(0, it) }
            val accepted = Array(weights.size) { IntArray(2) }
            for (i in 0 until n) {
                val instance = BitArray(weights.size, IntArray(1) { i })
                var sum = 0
                for (j in instance)
                    sum += weights[j]
                if (relation.violations(sum, degree) == 0) {
                    for (j in instance.indices) {
                        if (instance.isSet(j)) accepted[j][1] = 1
                        else accepted[j][0] = 1
                    }
                }
            }
            val canBeUnit = accepted.any { it[0] != it[1] }
            // We miss some cases, so this is only an implication (not an equivalence)
            if (relation.isUnit(lowerBound, upperBound, degree))
                assertTrue(canBeUnit)

            var nbrUnit = 0
            for (a in accepted) if (a[0] != a[1]) nbrUnit++

            val unitLiterals = IntArray(nbrUnit)
            var k = 0
            for ((ix, a) in accepted.withIndex()) if (a[0] != a[1]) unitLiterals[k++] =
                ix.toLiteral(a[0] == 0)

            val c = if (lowerBound == 0) {
                if (degree < 0) return
                Cardinality(IntRangeCollection(1, 4), degree, relation)
            } else Linear(IntHashMap().apply {
                this[1] = 0; this[2] = 1; this[3] = 2; this[4] = 3
            }, weights, degree, relation)

            if (c.isUnit()) {
                val actual = c.unitLiterals().apply { sort() }
                val expected = unitLiterals.apply { sort() }
                assertContentEquals(expected, actual)
            }
        }
        for (rel in entries) {
            for (degree in -1..5)
                test(IntArray(4) { 1 }, rel, degree)
        }
        for (rel in entries) {
            for (degree in -4..2)
                test(IntArray(4) { it - 2 }, rel, degree)
        }
    }
}
