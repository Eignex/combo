package combo.decisions

import com.eignex.klause.ast.BoolSpec
import com.eignex.klause.solver.localsearch.LocalSearchParams
import com.eignex.klause.solver.localsearch.LocalSearchSolver
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

private class WithMultiples : DecisionSpace() {
    val genres by multiple("Action", "Drama", "Horror")
    val premium by boolVar()
    val mustHaveAction by constraint { genres.contains("Action") }
    val atMostTwoGenres by constraint { genres.sizeLe(2) }
}

private class NestedMultiples : DecisionSpace() {
    val movies by decisionSpace(::Movies)
}

private class Movies : SubSpace() {
    val horror by multiple("Slasher", "Splatter", "Zombie")
}

class MultipleTest {

    @Test
    fun `multiple registers one bool per label under a dotted prefix`() {
        val space = WithMultiples().compileSpace()
        val boolIds = space.compiled.boolVarIdByName

        // Each label becomes its own bool variable.
        assertTrue("genres.Action" in boolIds)
        assertTrue("genres.Drama" in boolIds)
        assertTrue("genres.Horror" in boolIds)
        // Sibling decision variable still there.
        assertTrue("premium" in boolIds)
    }

    @Test
    fun `multiple appears in the definition as a single multiples entry`() {
        val def = WithMultiples().definition()
        assertEquals(mapOf("genres" to listOf("Action", "Drama", "Horror")), def.multiples)
        // The label-indicator bools should NOT also appear in variables — they live
        // exclusively under the multiples grouping.
        assertTrue("genres.Action" !in def.variables)
        // Sibling bool stays in variables.
        assertEquals(BoolSpec, def.variables["premium"])
    }

    @Test
    fun `constraint via contains forces the label indicator on every sample`() {
        val space = WithMultiples().compileSpace()
        val solver = LocalSearchSolver(space.compiled.problem)
        val actionBit = space.compiled.boolVarIdByName.getValue("genres.Action")
        // Any feasible sample must have genres.Action == true under the constraint.
        repeat(8) { seed ->
            val s = solver.sample(LocalSearchParams(randomSeed = seed.toLong())).assignment!!
            assertTrue(s.bools[actionBit], "sample $seed violated genres.contains(\"Action\")")
        }
    }

    @Test
    fun `sizeLe constraint bounds the selected-count`() {
        val space = WithMultiples().compileSpace()
        val solver = LocalSearchSolver(space.compiled.problem)
        val ids = listOf("Action", "Drama", "Horror")
            .map { space.compiled.boolVarIdByName.getValue("genres.$it") }
        repeat(8) { seed ->
            val s = solver.sample(LocalSearchParams(randomSeed = seed.toLong())).assignment!!
            val selected = ids.count { s.bools[it] }
            assertTrue(selected <= 2, "selected=$selected exceeded sizeLe(2)")
        }
    }

    @Test
    fun `nested multiple qualifies the indicator names under the parent path`() {
        val space = NestedMultiples().compileSpace()
        val boolIds = space.compiled.boolVarIdByName
        assertTrue("movies.horror.Slasher" in boolIds)
        assertTrue("movies.horror.Zombie" in boolIds)

        // And the def reflects the multiple at the nested level — not at the root.
        val def = NestedMultiples().definition()
        assertTrue(def.multiples.isEmpty(), "root should have no multiples; have ${def.multiples}")
        val nested = def.spaces.getValue("movies")
        assertEquals(listOf("Slasher", "Splatter", "Zombie"), nested.multiples["horror"])
    }

    @Test
    fun `contains rejects labels outside the declared set`() {
        val model = WithMultiples()
        model.compileSpace()
        assertFailsWith<IllegalArgumentException> { model.genres.contains("Unknown") }
    }
}
