package combo.decisions

import com.eignex.klause.ast.BoolSpec
import com.eignex.klause.ast.IntSpec
import com.eignex.klause.solver.LocalSearchParams
import com.eignex.klause.solver.LocalSearchSolver
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

private class CodeFeature : SubSpace() {
    val flag by boolVar()
    val budget by intVar(0, 100)
}

private class WithOneSubModel : DecisionSpace() {
    val gate by boolVar()
    val area by submodel(::CodeFeature)
}

private class Inner : SubSpace() {
    val toggle by boolVar()
}

private class Middle : SubSpace() {
    val knob by intVar(-5, 5)
    val inner by submodel(::Inner)
}

private class Nested : DecisionSpace() {
    val middle by submodel(::Middle)
}

private class AdSlot : SubSpace() {
    val premium by boolVar()
    val budget by intVar(0, 1000)
}

private class TwoAdSlots : DecisionSpace() {
    val slotA by submodel(::AdSlot)
    val slotB by submodel(::AdSlot)
}

class SubSpaceTest {

    @Test
    fun subModelVariablesGetQualifiedKlauseNames() {
        val space = WithOneSubModel().compileSpace()
        val entries = space.schemaDef.entries

        // Root-level decision is bare-named.
        assertTrue("gate" in entries, "expected 'gate' at root; have ${entries.keys}")

        // Sub-model decisions are prefixed by the property name they were mounted under.
        assertTrue("area.flag" in entries, "expected 'area.flag'; have ${entries.keys}")
        assertTrue("area.budget" in entries, "expected 'area.budget'; have ${entries.keys}")

        // Types are preserved through the prefix.
        assertTrue(entries["area.flag"] is BoolSpec)
        val budget = entries["area.budget"]
        assertTrue(budget is IntSpec)
        budget as IntSpec
        assertEquals(0, budget.min)
        assertEquals(100, budget.max)
    }

    @Test
    fun multipleInstancesOfSameSubModelGetDistinctNamespaces() {
        val model = TwoAdSlots()
        val space = model.compileSpace()
        val entries = space.schemaDef.entries

        // Both instances exist independently under their property-name prefix.
        assertTrue("slotA.premium" in entries)
        assertTrue("slotA.budget" in entries)
        assertTrue("slotB.premium" in entries)
        assertTrue("slotB.budget" in entries)

        // Their handles point at different klause variables.
        assertEquals("slotA.premium", model.slotA.premium.name)
        assertEquals("slotB.premium", model.slotB.premium.name)
        assertTrue(model.slotA.premium !== model.slotB.premium)
    }

    @Test
    fun nestedSubModelsAccumulateDottedPrefix() {
        val model = Nested()
        val space = model.compileSpace()
        val entries = space.schemaDef.entries
        assertTrue("middle.knob" in entries)
        assertTrue("middle.inner.toggle" in entries)
        assertEquals("middle.inner.toggle", model.middle.inner.toggle.name)
        assertEquals("middle.knob", model.middle.knob.name)
    }

    @Test
    fun subModelHandlesCarryQualifiedNamesEndToEnd() {
        val model = WithOneSubModel()
        val space = model.compileSpace()

        // The handle the user holds is the same one klause sees: qualified name.
        assertEquals("gate", model.gate.name)
        assertEquals("area.flag", model.area.flag.name)
        assertEquals("area.budget", model.area.budget.name)

        // Solver sees the sub-model's variables under the qualified names too.
        val solver = LocalSearchSolver(space.compiled.problem)
        val sample = solver.sample(LocalSearchParams(randomSeed = 1L))
        assertNotNull(sample, "solver must produce a feasible sample over a flat-bool problem")
        // Two top-level bools (gate, area.flag) + zero ints + one int from area.budget.
        assertEquals(2, sample.bools.size)
        assertEquals(1, sample.ints.size)
        assertTrue(sample.ints[0] in 0..100)

        // Decode by handle.
        space.compiled.decode(model.area.flag, sample)  // should not throw
        val b = space.compiled.decode(model.area.budget, sample)
        assertTrue(b in 0..100)
    }
}
