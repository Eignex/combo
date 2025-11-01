package combo.sat

import combo.math.nextBinomial
import combo.math.permutation
import combo.expressions.Constraint
import combo.model.TestModels
import combo.expressions.Cardinality
import combo.expressions.Conjunction
import combo.expressions.Disjunction
import combo.expressions.Relation
import combo.sat.optimizers.ExhaustiveSolver
import combo.test.assertContentEquals
import combo.util.IntArrayList
import combo.util.IntHashSet
import combo.util.isNotEmpty
import kotlin.random.Random
import kotlin.test.*

class ProblemTest {
    private val problem = Problem(
            6, arrayOf(Disjunction(IntArrayList(intArrayOf(1, 2, 3))),
            Disjunction(IntArrayList(intArrayOf(-1, -3))),
            Disjunction(IntArrayList(intArrayOf(5, 6))),
            Disjunction(IntArrayList(intArrayOf(-1, -2, -3, -5))),
            Conjunction(IntArrayList(intArrayOf(3))),
            Cardinality(IntArrayList(intArrayOf(3, 4)), 1, Relation.LE)
    ))

    @Test
    fun unitPropagationReduction() {
        val units = IntHashSet()
        val reduced = problem.unitPropagation(units)
        assertTrue(reduced.size < problem.nbrConstraints)
    }

    @Test
    fun unitPropagationSameSolution() {
        val solutions1 = ExhaustiveSolver(problem).asSequence().toSet()
        val units = IntHashSet()
        var reducedConstraints = problem.unitPropagation(units, returnConstraints = true)
        if (units.isNotEmpty()) reducedConstraints += Conjunction(units)
        val reducedProblem = Problem(problem.nbrValues, reducedConstraints)
        val solutions2 = ExhaustiveSolver(reducedProblem).asSequence(units).toSet()
        val unitsSentence = Conjunction(units)
        val constraints: MutableList<Constraint> = reducedProblem.constraints.toMutableList()
        constraints.add(unitsSentence)
        val solutions3 = ExhaustiveSolver(
                Problem(6, constraints.toTypedArray())).asSequence().toSet()
        for (l in solutions1) {
            assertTrue(problem.satisfies(l))
            assertTrue(reducedProblem.satisfies(l))
            assertTrue(solutions2.contains(l))
            assertTrue(solutions3.contains(l))
        }
        assertEquals(solutions1.size, solutions2.size)
        assertEquals(solutions1.size, solutions3.size)
    }

    @Test
    fun unitPropagationUnsat() {
        assertFailsWith(UnsatisfiableException::class) {
            Problem(1, arrayOf(Disjunction(IntArrayList(intArrayOf(1))), Disjunction(IntArrayList(intArrayOf(-1))))).unitPropagation()
        }
    }

    @Test
    fun randomPropagation() {
        val rng = Random.Default
        val p = TestModels.LARGE2.problem
        val perm = permutation(p.nbrValues, rng)
        val lits = (0 until rng.nextBinomial(0.7f, p.nbrValues)).asSequence()
                .map { perm.encode(it) }
                .map { it.toLiteral(rng.nextBoolean()) }
                .toList().toIntArray().apply { sort() }
        val constraints: Array<Constraint> = p.constraints.toList().toTypedArray()
        val p2 = Problem(p.nbrValues, constraints + Conjunction(IntArrayList(lits)))
        val reduced = try {
            val units = IntHashSet().apply { addAll(lits) }
            var reduced = p.unitPropagation(units)
            if (units.isNotEmpty()) reduced += Conjunction(units)
            Problem(p.nbrValues, reduced)
        } catch (e: UnsatisfiableException) {
            return
        }
        InstancePermutation(p.nbrValues, BitArrayFactory, rng).iterator().asSequence().take(100).forEach {
            assertEquals(p2.satisfies(it), reduced.satisfies(it))
        }
    }

    @Test
    fun satisfies() {
        val constraints = arrayOf(Cardinality(IntArrayList(intArrayOf(1, 2, 3)), 1, Relation.LE))
        val problem = Problem(3, constraints)
        assertFalse(problem.satisfies(BitArray(3, IntArray(1) { 0b110 })))
        assertTrue(problem.satisfies(BitArray(3, IntArray(1) { 0b000 })))
        assertTrue(problem.satisfies(BitArray(3, IntArray(1) { 0b010 })))
    }

    @Test
    fun clauseMatch() {
        val constraints: Array<out Constraint> = arrayOf(
                Disjunction(IntArrayList(intArrayOf(1, 2, 3))),
                Conjunction(IntArrayList(intArrayOf(-1))))
        val problem = Problem(3, constraints)
        assertContentEquals(intArrayOf(0, 1), problem.constraining(0))
        assertContentEquals(intArrayOf(0), problem.constraining(1))
        assertContentEquals(intArrayOf(0), problem.constraining(2))
    }
}
