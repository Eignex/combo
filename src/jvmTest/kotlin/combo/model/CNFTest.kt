package combo.model

import combo.expressions.Disjunction
import combo.expressions.Root
import combo.test.assertContentEquals
import combo.util.collectionOf
import kotlin.test.Test
import kotlin.test.assertEquals

class CNFTest {

    @Test
    fun pullIn() {
        // (a V b) and (c V d)
        // V e V f
        // = (a V b V e V f) and (a V b V e V f)
        val cnf1 = CNF(listOf(Disjunction(collectionOf(1, 2)), Disjunction(collectionOf(3, 4))))
        val cnf2 = cnf1.pullIn(Disjunction(collectionOf(5, 6)))
        assertEquals(2, cnf1.disjunctions.size)
        assertEquals(2, cnf2.disjunctions.size)
        assertContentEquals(intArrayOf(1, 2, 5, 6), cnf2.disjunctions[0].literals.toArray().apply { sort() })
        assertContentEquals(intArrayOf(3, 4, 5, 6), cnf2.disjunctions[1].literals.toArray().apply { sort() })
    }

    @Test
    fun pullInDuplication() {
        // (a V b) and (c V d)
        // V a V c
        // = (a V b V c) and (a V c V d)
        val cnf1 = CNF(listOf(Disjunction(collectionOf(1, 2)), Disjunction(collectionOf(3, 4))))
        val cnf2 = cnf1.pullIn(Disjunction(collectionOf(1, 3)))
        assertContentEquals(intArrayOf(1, 2, 3), cnf2.disjunctions[0].literals.toArray().apply { sort() })
        assertContentEquals(intArrayOf(1, 3, 4), cnf2.disjunctions[1].literals.toArray().apply { sort() })
    }

    @Test
    fun pullInTautology() {
        // (a V b) and (c V d)
        // V !a
        // = (!a V c V d)
        val cnf1 = CNF(listOf(Disjunction(collectionOf(1, 2)), Disjunction(collectionOf(3, 4))))
        val cnf2 = cnf1.pullIn(Disjunction(collectionOf(-1)))
        assertEquals(1, cnf2.disjunctions.size)
        assertContentEquals(intArrayOf(-1, 3, 4), cnf2.disjunctions[0].literals.toArray().apply { sort() })
    }

    @Test
    fun distribute() {
        // (a V b) and (c V d)
        // or (!a V e V f) and (!b V d V f)
        // = (!a V c V d V e V f) and (!b V c V d V f)
        val cnf1 = CNF(listOf(Disjunction(collectionOf(1, 2)), Disjunction(collectionOf(3, 4))))
        val cnf2 = CNF(listOf(Disjunction(collectionOf(-1, 5, 6)), Disjunction(collectionOf(-2, 4, 6))))
        ConstraintFactory(RootScope(Root("")), VariableIndex()).and(cnf1, cnf2)
        val cnf3 = cnf1.distribute(cnf2)
        assertContentEquals(intArrayOf(-1, 3, 4, 5, 6), cnf3.disjunctions[0].literals.toArray().apply { sort() })
        assertContentEquals(intArrayOf(-2, 3, 4, 6), cnf3.disjunctions[1].literals.toArray().apply { sort() })
    }
}
