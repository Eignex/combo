package combo.sat

import combo.math.permutation
import combo.model.Model
import combo.expressions.Tautology
import combo.expressions.Cardinality
import combo.test.assertContentEquals
import combo.util.mapArray
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class TransitiveImplicationsTest {

    @Test
    fun unsatisfiedByImplicationDigraph() {
        val problem = Model.model("Simple 2-Unsat") {
            val f1 = bool()
            val f2 = bool()
            impose { f1 or f2 }
            impose { f1 or !f2 }
            impose { !f1 or f2 }
            impose { !f1 or !f2 }
        }.problem

        assertFailsWith(UnsatisfiableException::class) { TransitiveImplications(problem) }
    }

    @Test
    fun simpleGraph() {
        val id = TransitiveImplications(10, mapOf(
                1 to intArrayOf(2, 6, 8),
                2 to intArrayOf(3, 1),
                3 to intArrayOf(4, 2),
                4 to intArrayOf(5),
                5 to intArrayOf(4),
                6 to intArrayOf(7),
                7 to intArrayOf(6, 4),
                8 to intArrayOf(9),
                9 to intArrayOf(8, 10, 5, 3)
        ))

        assertContentEquals(intArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), id.toArray(1).apply { sort() })
        assertContentEquals(intArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), id.toArray(2).apply { sort() })
        assertContentEquals(intArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), id.toArray(3).apply { sort() })
        assertContentEquals(intArrayOf(4, 5), id.toArray(4).apply { sort() })
        assertContentEquals(intArrayOf(4, 5), id.toArray(5).apply { sort() })
        assertContentEquals(intArrayOf(4, 5, 6, 7), id.toArray(6).apply { sort() })
        assertContentEquals(intArrayOf(4, 5, 6, 7), id.toArray(7).apply { sort() })
        assertContentEquals(intArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), id.toArray(8).apply { sort() })
        assertContentEquals(intArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), id.toArray(9).apply { sort() })
        assertContentEquals(intArrayOf(), id.toArray(10).apply { sort() })
    }

    @Test
    fun doubleLinearChainGraph() {
        val problem = Model.model("Flag chain") {
            var next: Model.ModelBuilder<*> = this
            for (k in 1..100)
                next = next.model("$k") {}
        }.problem

        val id = TransitiveImplications(problem)

        for (i in 0 until problem.nbrValues) {
            assertContentEquals((1 until (i + 1)).toList().toIntArray(),
                    id.toArray(i.toLiteral(true)).apply { sort() })
            assertContentEquals(((i + 2)..problem.nbrValues).toList().toIntArray().mapArray { -it }.apply { sort() },
                    id.toArray(i.toLiteral(false)).apply { sort() })
        }
    }

    @Test
    fun sat2Digraph() {
        val problem = Model.model {
            bool("x1")
            model("x2") {
                bool("x3")
            }
            bool("x4")
            impose { "x2" or "x4" }
        }.problem
        val id = TransitiveImplications(problem)
        for (i in 0 until 10) {
            val instance = BitArray(problem.nbrValues).also { RandomSet().initialize(it,
                Tautology, Random, null) }
            for (j in permutation(problem.nbrValues, Random)) {
                val lit = instance.literal(j)
                id.trueImplications(lit)?.run { instance.or(this) }
                id.falseImplications(lit)?.run { instance.andNot(this) }
            }
            assertTrue(problem.satisfies(instance))
        }
    }

    @Test
    fun cardinalityExclusiveDigraph() {
        val problem = Model.model {
            nominal(values = Array(100) { it })
        }.problem
        val card = problem.constraints.first { it is Cardinality }
        val id = TransitiveImplications(problem)

        for (i in 0 until 10) {
            val instance = BitArray(problem.nbrValues).also { RandomSet().initialize(it,
                Tautology, Random, null) }
            for (j in permutation(problem.nbrValues, Random)) {
                val lit = instance.literal(j)
                id.trueImplications(lit)?.run { instance.or(this) }
                id.falseImplications(lit)?.run { instance.andNot(this) }
            }
            assertTrue(card.satisfies(instance))
        }
    }
}
