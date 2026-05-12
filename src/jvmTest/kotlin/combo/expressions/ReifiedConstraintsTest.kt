package combo.expressions

import combo.sat.*
import combo.sat.optimizers.ExhaustiveSolver
import combo.test.assertContentEquals
import combo.util.IntArrayList
import combo.util.collectionOf
import kotlin.random.Random
import kotlin.test.*

class ReifiedEquivalentTest : ConstraintTest() {

    @Test
    fun violationsDisjunction() {
        val d =
            ReifiedEquivalent(-1, Disjunction(IntArrayList(intArrayOf(2, -3))))
        assertEquals(1, d.violations(BitArray(3, IntArray(1) { 0b111 })))
        assertEquals(1, d.violations(BitArray(3, IntArray(1) { 0b011 })))
        assertEquals(0, d.violations(BitArray(3, IntArray(1) { 0b101 })))
        assertEquals(1, d.violations(BitArray(3, IntArray(1) { 0b001 })))
        assertEquals(0, d.violations(BitArray(3, IntArray(1) { 0b110 })))
        assertEquals(0, d.violations(BitArray(3, IntArray(1) { 0b010 })))
        assertEquals(1, d.violations(BitArray(3, IntArray(1) { 0b100 })))
        assertEquals(0, d.violations(BitArray(3, IntArray(1) { 0b000 })))
    }

    @Test
    fun violationsConjunction() {
        val d =
            ReifiedEquivalent(1, Conjunction(IntArrayList(intArrayOf(2, -3))))
        assertEquals(1, d.violations(BitArray(3, IntArray(1) { 0b111 })))
        assertEquals(0, d.violations(BitArray(3, IntArray(1) { 0b011 })))
        assertEquals(1, d.violations(BitArray(3, IntArray(1) { 0b101 })))
        assertEquals(1, d.violations(BitArray(3, IntArray(1) { 0b001 })))
        assertEquals(0, d.violations(BitArray(3, IntArray(1) { 0b110 })))
        assertEquals(1, d.violations(BitArray(3, IntArray(1) { 0b010 })))
        assertEquals(0, d.violations(BitArray(3, IntArray(1) { 0b100 })))
        assertEquals(0, d.violations(BitArray(3, IntArray(1) { 0b000 })))
    }

    @Test
    fun updateCache() {
        fun testUpdateCache(c: PropositionalConstraint) {
            val r = ReifiedEquivalent(-1, c)
            for (k in 0 until 16) {
                val instance = BitArray(4, IntArray(1) { k })
                randomCacheUpdates(instance, r)
            }
        }
        for (c in arrayOf(
            Conjunction(collectionOf(2, -4)),
            Disjunction(collectionOf(2, 3)),
            Disjunction(collectionOf(-2, -3)),
            Cardinality(collectionOf(2, 3, 4), 2, Relation.EQ),
            Cardinality(collectionOf(2, 3, 4), 1, Relation.LE),
            Cardinality(collectionOf(2, 3, 4), 1, Relation.NE),
            ReifiedEquivalent(-2, Disjunction(collectionOf(3, -4)))
        ))
            testUpdateCache(c)
    }

    @Test
    fun toCnfConjunction() {
        val e =
            ReifiedEquivalent(-1, Conjunction(IntArrayList(intArrayOf(2, 3))))
        val c = e.toCnf().toList<Constraint>().toTypedArray()
        assertEquals(3, c.size)
        val s1 = ExhaustiveSolver(Problem(3, arrayOf(e))).asSequence().toSet()
        val s2 = ExhaustiveSolver(Problem(3, c)).asSequence().toSet()
        assertEquals(s1.size, s2.size)
        for (l in s1) assertTrue(s2.contains(l))
    }

    @Test
    fun toCnfDisjunction() {
        val e =
            ReifiedEquivalent(2, Disjunction(IntArrayList(intArrayOf(1, -3))))
        val c = e.toCnf().toList<Constraint>().toTypedArray()
        assertEquals(3, c.size)
        val s1 = ExhaustiveSolver(Problem(3, arrayOf(e))).asSequence().toList()
        val s2 = ExhaustiveSolver(Problem(3, c)).asSequence().toList()
        assertEquals(s1.size, s2.size)
        for (l in s1) assertTrue(s2.contains(l))
    }

    @Test
    fun toCnfSatisfiesDisjunction() {
        val original =
            ReifiedEquivalent(-3, Disjunction(IntArrayList(intArrayOf(-1, -2))))
        val toCnf = original.toCnf()
        for (l in InstancePermutation(3, BitArrayFactory, Random)) {
            val s1 = original.satisfies(l)
            val s2 = toCnf.asSequence().all { it.satisfies(l) }
            assertEquals(s1, s2)
        }
    }

    @Test
    fun toCnfSatisfiesConjunction() {
        val original =
            ReifiedEquivalent(3, Conjunction(IntArrayList(intArrayOf(1, -2))))
        val toCnf = original.toCnf()
        for (l in InstancePermutation(3, BitArrayFactory, Random)) {
            val s1 = original.satisfies(l)
            val s2 = toCnf.asSequence().all { it.satisfies(l) }
            assertEquals(s1, s2)
        }
    }

    @Test
    fun propagateUnitReturnsReifiedConstraint() {
        val r = ReifiedEquivalent(
            3,
            Disjunction(IntArrayList(intArrayOf(-1, -2, -4)))
        )
        val clause = r.unitPropagation(3)
        assertTrue(clause is Disjunction)
        assertContentEquals(
            intArrayOf(-1, -2, -4),
            clause.literals.sortedBy { it.toIx() }.toIntArray()
        )
    }

    @Test
    fun propagateNegUnitNegatesClauseDisjunction() {
        val r = ReifiedEquivalent(
            3,
            Disjunction(IntArrayList(intArrayOf(-1, -2, -4)))
        )
        val clause = r.unitPropagation(-3)
        assertTrue(clause is Conjunction)
        assertContentEquals(
            intArrayOf(1, 2, 4),
            clause.literals.sortedBy { it.toIx() }.toIntArray()
        )
    }

    @Test
    fun propagateNegUnitNegatesClauseConjunction() {
        val r = ReifiedEquivalent(
            4,
            Conjunction(IntArrayList(intArrayOf(-1, -2, -3)))
        )
        val clause = r.unitPropagation(-4)
        assertTrue(clause is Disjunction)
        assertContentEquals(
            intArrayOf(1, 2, 3),
            clause.literals.sortedBy { it.toIx() }.toIntArray()
        )
    }

    @Test
    fun propagatePosUnitConjunction() {
        val r = ReifiedEquivalent(
            2,
            Conjunction(IntArrayList(intArrayOf(1, 3, 4, 5)))
        )
        val s = r.unitPropagation(5)
        assertContentEquals(
            intArrayOf(1, 3, 4),
            (s as ReifiedEquivalent).constraint.literals.sortedBy { it.toIx() }
                .toIntArray()
        )
    }

    @Test
    fun propagatePosUnitDisjunction() {
        val r = ReifiedEquivalent(
            2,
            Disjunction(IntArrayList(intArrayOf(1, 3, 4, 5)))
        )
        val s = r.unitPropagation(5)
        assertContentEquals(
            intArrayOf(2),
            (s as Conjunction).literals.toArray()
        )
    }

    @Test
    fun propagateLastNegConjunction() {
        val r =
            ReifiedEquivalent(1, Conjunction(IntArrayList(intArrayOf(2, 3))))
        val c = r.unitPropagation(5).unitPropagation(-2)
        assertContentEquals(
            intArrayOf(-1),
            (c as Conjunction).literals.toArray()
        )
    }

    @Test
    fun propagateLastPosConjunction() {
        val r =
            ReifiedEquivalent(1, Conjunction(IntArrayList(intArrayOf(2, 3))))
        val c = r.unitPropagation(2).unitPropagation(3)
        assertContentEquals(
            intArrayOf(1),
            (c as Conjunction).literals.toArray()
        )
    }

    @Test
    fun propagateLastNegDisjunction() {
        val r =
            ReifiedEquivalent(1, Disjunction(IntArrayList(intArrayOf(2, 3))))
        val c = r.unitPropagation(-3).unitPropagation(-2)
        assertContentEquals(
            intArrayOf(-1),
            (c as Conjunction).literals.toArray()
        )
    }

    @Test
    fun propagateLastPosDisjunction() {
        val r =
            ReifiedEquivalent(1, Disjunction(IntArrayList(intArrayOf(2, 3))))
        val c = r.unitPropagation(2).unitPropagation(3)
        assertContentEquals(
            intArrayOf(1),
            (c as Conjunction).literals.toArray()
        )
    }

    @Test
    fun randomExhaustivePropagations() {
        randomExhaustivePropagations(
            ReifiedEquivalent(
                1,
                Conjunction(collectionOf(2, 3, 4, 5))
            )
        )
        randomExhaustivePropagations(
            ReifiedEquivalent(
                5,
                Conjunction(collectionOf(-1, -2, 3, -4))
            )
        )
        randomExhaustivePropagations(
            ReifiedEquivalent(
                2,
                Conjunction(collectionOf(-3, -4))
            )
        )
        randomExhaustivePropagations(
            ReifiedEquivalent(
                1,
                Disjunction(collectionOf(2, 3, 4, 5))
            )
        )
        randomExhaustivePropagations(
            ReifiedEquivalent(
                2,
                Disjunction(collectionOf(1, 3, 4, 5))
            )
        )
        randomExhaustivePropagations(
            ReifiedEquivalent(
                1,
                Cardinality(collectionOf(2, 3, 4, 5), 2, Relation.EQ)
            )
        )
        randomExhaustivePropagations(
            ReifiedEquivalent(
                2,
                Cardinality(collectionOf(1, 3, 4, 5), 1, Relation.GE)
            )
        )
    }

    @Test
    fun randomCoerce() {
        randomCoerce(ReifiedEquivalent(2, Disjunction(collectionOf(1, 4, 5))))
        randomCoerce(ReifiedEquivalent(-2, Disjunction(collectionOf(1, 4, 5))))
        randomCoerce(ReifiedEquivalent(3, Disjunction(collectionOf(1, -4, 5))))
        randomCoerce(ReifiedEquivalent(-2, Disjunction(collectionOf(1, 5))))
        randomCoerce(ReifiedEquivalent(2, Disjunction(collectionOf(1, -5))))
        randomCoerce(ReifiedEquivalent(1, Disjunction(collectionOf(-3))))
        randomCoerce(ReifiedEquivalent(-2, Conjunction(collectionOf(1, 5))))
        randomCoerce(ReifiedEquivalent(2, Conjunction(collectionOf(1, -5))))
        randomCoerce(
            ReifiedEquivalent(
                -4,
                Cardinality(collectionOf(1, 2, 3), 2, Relation.EQ)
            )
        )
    }
}

class ReifiedImpliesTest : ConstraintTest() {

    @Test
    fun violationsDisjunction() {
        val d = ReifiedImplies(-1, Disjunction(IntArrayList(intArrayOf(2, -3))))
        assertEquals(0, d.violations(BitArray(3, IntArray(1) { 0b111 })))
        assertEquals(0, d.violations(BitArray(3, IntArray(1) { 0b011 })))
        assertEquals(0, d.violations(BitArray(3, IntArray(1) { 0b101 })))
        assertEquals(0, d.violations(BitArray(3, IntArray(1) { 0b001 })))
        assertEquals(0, d.violations(BitArray(3, IntArray(1) { 0b110 })))
        assertEquals(0, d.violations(BitArray(3, IntArray(1) { 0b010 })))
        assertEquals(1, d.violations(BitArray(3, IntArray(1) { 0b100 })))
        assertEquals(0, d.violations(BitArray(3, IntArray(1) { 0b000 })))
    }

    @Test
    fun violationsConjunction() {
        val d = ReifiedImplies(1, Conjunction(IntArrayList(intArrayOf(2, -3))))
        assertEquals(1, d.violations(BitArray(3, IntArray(1) { 0b111 })))
        assertEquals(0, d.violations(BitArray(3, IntArray(1) { 0b011 })))
        assertEquals(1, d.violations(BitArray(3, IntArray(1) { 0b101 })))
        assertEquals(1, d.violations(BitArray(3, IntArray(1) { 0b001 })))
        assertEquals(0, d.violations(BitArray(3, IntArray(1) { 0b110 })))
        assertEquals(0, d.violations(BitArray(3, IntArray(1) { 0b010 })))
        assertEquals(0, d.violations(BitArray(3, IntArray(1) { 0b100 })))
        assertEquals(0, d.violations(BitArray(3, IntArray(1) { 0b000 })))
    }

    @Test
    fun updateCache() {
        fun testUpdateCache(c: Constraint) {
            val r = ReifiedImplies(-1, c)
            for (k in 0 until 16) {
                val instance = BitArray(4, IntArray(1) { k })
                randomCacheUpdates(instance, r)
            }
        }
        for (c in arrayOf(
            Conjunction(collectionOf(2, -4)),
            Disjunction(collectionOf(2, 3)),
            Disjunction(collectionOf(-2, -3)),
            Cardinality(collectionOf(2, 3, 4), 2, Relation.EQ),
            Cardinality(collectionOf(2, 3, 4), 1, Relation.LE),
            Cardinality(collectionOf(2, 3, 4), 1, Relation.NE),
            ReifiedImplies(-2, Disjunction(collectionOf(3, -4)))
        ))
            testUpdateCache(c)
    }

    @Test
    fun propagateUnitReturnsReifiedConstraint() {
        val r =
            ReifiedImplies(3, Disjunction(IntArrayList(intArrayOf(-1, -2, -4))))
        val clause = r.unitPropagation(3)
        assertTrue(clause is Disjunction)
        assertContentEquals(
            intArrayOf(-1, -2, -4),
            clause.literals.sortedBy { it.toIx() }.toIntArray()
        )
    }

    @Test
    fun propagateNegUnitNegatesClause() {
        val r =
            ReifiedImplies(3, Disjunction(IntArrayList(intArrayOf(-1, -2, -4))))
        val clause = r.unitPropagation(-3)
        assertTrue(clause is Tautology)
    }

    @Test
    fun propagatePosUnitConjunction() {
        val r =
            ReifiedImplies(2, Conjunction(IntArrayList(intArrayOf(1, 3, 4, 5))))
        val s = r.unitPropagation(5)
        assertContentEquals(
            intArrayOf(1, 3, 4),
            (s as ReifiedImplies).constraint.literals.sortedBy { it.toIx() }
                .toIntArray()
        )
    }

    @Test
    fun propagatePosUnitDisjunction() {
        val r =
            ReifiedImplies(2, Disjunction(IntArrayList(intArrayOf(1, 3, 4, 5))))
        val s = r.unitPropagation(5)
        assertTrue(s is Tautology)
    }

    @Test
    fun propagateLastNegConjunction() {
        val r = ReifiedImplies(1, Conjunction(IntArrayList(intArrayOf(2, 3))))
        val c = r.unitPropagation(5).unitPropagation(-2)
        assertContentEquals(
            intArrayOf(-1),
            (c as Conjunction).literals.toArray()
        )
    }

    @Test
    fun propagateLastPosConjunction() {
        val r = ReifiedImplies(1, Conjunction(IntArrayList(intArrayOf(2, 3))))
        val c = r.unitPropagation(2).unitPropagation(3)
        assertTrue(c is Tautology)
    }

    @Test
    fun propagateLastNegDisjunction() {
        val r = ReifiedImplies(1, Disjunction(IntArrayList(intArrayOf(2, 3))))
        val c = r.unitPropagation(-3).unitPropagation(-2)
        assertContentEquals(
            intArrayOf(-1),
            (c as Conjunction).literals.toArray()
        )
    }

    @Test
    fun propagateLastPosDisjunction() {
        val r = ReifiedImplies(1, Disjunction(IntArrayList(intArrayOf(2, 3))))
        val c = r.unitPropagation(2).unitPropagation(3)
        assertTrue(c is Tautology)
    }

    @Test
    fun randomExhaustivePropagations() {
        randomExhaustivePropagations(
            ReifiedImplies(
                1,
                Conjunction(collectionOf(2, 3, 4, 5))
            )
        )
        randomExhaustivePropagations(
            ReifiedImplies(
                5,
                Conjunction(collectionOf(-1, -2, 3, -4))
            )
        )
        randomExhaustivePropagations(
            ReifiedImplies(
                2,
                Conjunction(collectionOf(-3, -4))
            )
        )
        randomExhaustivePropagations(
            ReifiedImplies(
                1,
                Disjunction(collectionOf(2, 3, 4, 5))
            )
        )
        randomExhaustivePropagations(
            ReifiedImplies(
                2,
                Disjunction(collectionOf(1, 3, 4, 5))
            )
        )
        randomExhaustivePropagations(
            ReifiedImplies(
                1,
                Cardinality(collectionOf(2, 3, 4, 5), 2, Relation.EQ)
            )
        )
        randomExhaustivePropagations(
            ReifiedImplies(
                2,
                Cardinality(collectionOf(1, 3, 4, 5), 1, Relation.GE)
            )
        )
    }

    @Test
    fun randomCoerce() {
        randomCoerce(ReifiedImplies(2, Disjunction(collectionOf(1, 4, 5))))
        randomCoerce(ReifiedImplies(-2, Disjunction(collectionOf(1, 4, 5))))
        randomCoerce(ReifiedImplies(3, Disjunction(collectionOf(1, -4, 5))))
        randomCoerce(ReifiedImplies(-2, Disjunction(collectionOf(1, 5))))
        randomCoerce(ReifiedImplies(2, Disjunction(collectionOf(1, -5))))
        randomCoerce(ReifiedImplies(1, Disjunction(collectionOf(-3))))
        randomCoerce(ReifiedImplies(-2, Conjunction(collectionOf(1, 5))))
        randomCoerce(ReifiedImplies(2, Conjunction(collectionOf(1, -5))))
        randomCoerce(
            ReifiedImplies(
                -4,
                Cardinality(collectionOf(1, 2, 3), 2, Relation.EQ)
            )
        )
    }
}
