package combo.model

import combo.sat.BitArray
import combo.sat.BitArrayFactory
import combo.sat.InstancePermutation
import combo.expressions.Cardinality
import combo.expressions.Conjunction
import combo.expressions.Flag
import combo.expressions.Multiple
import combo.expressions.Nominal
import combo.expressions.Relation
import combo.sat.not
import combo.sat.optimizers.ExhaustiveSolver
import combo.util.collectionOf
import kotlin.random.Random
import kotlin.test.*

class ModelTest {
    @Test
    fun cantUseUnregisteredFeature() {
        assertFailsWith(NoSuchElementException::class) {
            Model.model {
                impose { "m1" and "m2" }
            }
        }
    }

    @Test
    fun cantRegisterFeatureTwice() {
        assertFailsWith(IllegalArgumentException::class) {
            Model.model {
                bool("f1")
                bool("f1")
            }
        }
    }

    @Test
    fun emptyModel() {
        val m1 = Model.model { }
        assertEquals(0, m1.problem.nbrValues)
    }

    @Test
    fun model1() {
        with(TestModels.MODEL1) {
            assertEquals(7, scope.asSequence().count())
            assertEquals(13, problem.nbrValues)

            val solver = ModelOptimizer(this, ExhaustiveSolver(problem))
            val assignments = solver.asSequence(this["f3"]).toList()
            assignments.forEach {
                assertTrue(it.contains("f2"))
                assertTrue(it.contains("f3"))
                assertTrue(it.contains("alt1"))
                assertEquals("d", it.getString("alt2"))
            }
        }
    }

    @Test
    fun model2() {
        with(TestModels.MODEL2) {
            assertEquals(5, scope.asSequence().count())
            assertEquals(10, problem.nbrValues)
            assertNotNull(problem.constraints.find { c -> c is Conjunction && c.literals.size == 10 })

            val solver = ModelOptimizer(this, ExhaustiveSolver(problem))
            val assignments = solver.asSequence().toList()
            assertEquals(1, assignments.size)
            assertEquals(2, assignments[0].toMap().size)
            assertTrue(assignments[0].getBoolean("f1"))
            assertTrue(assignments[0].getBoolean("f2"))
        }
    }

    @Test
    fun model3() {
        with(TestModels.MODEL3) {
            assertEquals(2, scope.asSequence().count())
            assertEquals(6, problem.nbrValues)
            assertNotNull(problem.constraints.find { c -> c is Conjunction && c.literals.size == 4 })

            val solver = ModelOptimizer(this, ExhaustiveSolver(problem))
            val assignments = solver.asSequence()
            assignments.forEach {
                assertEquals("c", it.getString("alt1"))
                assertTrue("b" in it.get<List<String>>("mult1")!!)
            }
        }
    }

    @Test
    fun model4() {
        with(TestModels.MODEL4) {
            assertEquals(5, scope.asSequence().count())
            assertEquals(15, problem.nbrValues)

            val solver = ModelOptimizer(this, ExhaustiveSolver(problem))
            val assignments = solver.asSequence(
                    scope.find<Nominal<Int>>("a1")!!.value(2),
                    !this["m1"])
            assignments.forEach {
                assertTrue(it.contains("a1"))
                assertFalse(it.contains("m1"))
            }
        }
    }

    @Test
    fun model5() {
        with(TestModels.MODEL5) {
            assertEquals(10, scope.asSequence().count())
            assertEquals("r1", scope.children.first().scopeName)
            assertEquals("sub1", scope.children.first().children.first().scopeName)
            assertEquals("sub2", scope.children.first().children.first().children.first().scopeName)
            assertEquals("sub3", scope.children.first().children.first().children.first().children.first().scopeName)
            assertEquals("sub4", scope.children.first().children.first().children.first().children.first().asSequence().first().name)
            assertEquals("r2", scope.children.last().scopeName)

            val solver = ModelOptimizer(this, ExhaustiveSolver(problem))
            val assignments = solver.asSequence()
            assignments.forEach {
                assertTrue(it.contains("f1") || it.contains("sub4"))
                if (it.contains("sub2")) assertTrue(it.contains("sub1"))
            }
        }
    }

    @Test
    fun model6() {
        with(TestModels.MODEL6) {
            assertEquals(12, scope.asSequence().count())
            assertEquals(6, problem.constraints.size)
            assertEquals(Relation.values().size - 1,
                    problem.constraints.filterIsInstance<Cardinality>().map { it.relation }.toSet().size)

            val solver = ModelOptimizer(this, ExhaustiveSolver(problem))
            val assignments = solver.asSequence(this["0"], this["2"], this["5"], this["8"])
            assignments.forEach {
                assertFalse(it.contains("1"))
            }
        }
    }

    @Test
    fun model7() {
        with(TestModels.MODEL7) {

            assertEquals(4, scope.asSequence().count())
            assertEquals(3, problem.constraints.size)

            val solver = ModelOptimizer(this, ExhaustiveSolver(this.problem))

            val assignments1 = solver.asSequence(this["f2"]).map { it.toMap() }.toList()
            assertEquals(1, assignments1.size)
            assignments1.forEach {
                assertTrue(it.containsKey(this["f1"]))
                assertTrue(it.containsKey(this["f2"]))
            }
            assertNull(solver.witness(this["f1"], !this["sub2"]))

            val assignments2 = solver.asSequence(this["f1"]).map { it.toMap() }.toList()
            assertEquals(2, assignments2.size)
            assignments2.forEach {
                assertTrue(it.containsKey(this["f1"]))
            }
        }
    }

    @Test
    fun unsat1() {
        with(TestModels.UNSAT1) {
            assertEquals(2, scope.asSequence().count())
            assertEquals(4, problem.constraints.size)
            assertNull(ExhaustiveSolver(problem).witness())
        }
    }

    @Test
    fun unsat2() {
        with(TestModels.UNSAT2) {
            assertEquals(3, scope.asSequence().count())
            assertEquals(8, problem.constraints.size)
            assertNull(ExhaustiveSolver(problem).witness())
        }
    }

    @Test
    fun unsat3() {
        with(TestModels.UNSAT3) {
            assertEquals(1, scope.asSequence().count())
            assertEquals(3, problem.constraints.size)
            assertNull(ExhaustiveSolver(problem).witness())
        }
    }

    @Test
    fun unsat4() {
        with(TestModels.UNSAT4) {
            assertEquals(3, scope.asSequence().count())
            assertEquals(6, problem.constraints.size)
            assertNull(ExhaustiveSolver(problem).witness())
        }
    }

    @Test
    fun large1() {
        with(TestModels.LARGE1) {
            assertEquals(4, scope.asSequence().count())
            assertEquals(6, problem.constraints.size)
            assertEquals(322, problem.nbrValues)
        }
    }

    @Test
    fun large2() {
        with(TestModels.LARGE2) {
            assertEquals(55, scope.asSequence().count())
            assertEquals(107, problem.constraints.size)
            assertEquals(555, problem.nbrValues)

            val m1: Flag<*> = scope.find("Cat 1")!!
            val m2: Multiple<*> = scope.find("Cat 1 1")!!
            val con = Conjunction(collectionOf(!m1.toLiteral(index), m2.toLiteral(index)))
            for (i in InstancePermutation(problem.nbrValues, BitArrayFactory, Random).asSequence().take(100)) {
                con.coerce(i, Random)
                assertFalse(problem.satisfies(i), "$i")
            }
        }
    }

    @Test
    fun large3() {
        with(TestModels.LARGE3) {
            assertEquals(500, scope.asSequence().count())
            assertEquals(499, problem.constraints.size)
            assertEquals(500, problem.nbrValues)
        }
    }

    @Test
    fun large4() {
        with(TestModels.LARGE4) {
            assertEquals(500, scope.asSequence().count())
            assertEquals(1000, problem.constraints.size)
            assertEquals(500, problem.nbrValues)
        }
    }

    @Test
    fun numeric1() {
        with(TestModels.NUMERIC1) {
            assertEquals(7, scope.asSequence().count())
            assertEquals(11, problem.constraints.size)
            assertEquals(104, problem.nbrValues)
            val instance = BitArray(problem.nbrValues)
            for (c in problem.constraints)
                c.coerce(instance, Random)
            assertTrue(problem.satisfies(instance))
        }
    }

    @Test
    fun numeric2() {
        with(TestModels.NUMERIC2) {
            assertEquals(3, scope.asSequence().count())
            assertEquals(5, problem.constraints.size)
            assertEquals(99, problem.nbrValues)
            val instance = BitArray(problem.nbrValues)
            for (c in problem.constraints)
                c.coerce(instance, Random)
            assertTrue(problem.satisfies(instance))
        }
    }

    @Test
    fun numeric3() {
        with(TestModels.NUMERIC3) {
            assertEquals(13, scope.asSequence().count())
            assertEquals(20, problem.constraints.size)
            assertEquals(408, problem.nbrValues)
            val instance = BitArray(problem.nbrValues)
            for (c in problem.constraints)
                c.coerce(instance, Random)
            assertTrue(problem.satisfies(instance))
        }
    }
}
