package combo.decisions

import com.eignex.klause.ast.BoolSpec
import com.eignex.klause.ast.IntSpec
import com.eignex.klause.solver.localsearch.LocalSearchParams
import com.eignex.klause.solver.localsearch.LocalSearchSolver
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
    val area by decisionSpace(::CodeFeature)
}

private class Inner : SubSpace() {
    val toggle by boolVar()
}

private class Middle : SubSpace() {
    val knob by intVar(-5, 5)
    val inner by decisionSpace(::Inner)
}

private class Nested : DecisionSpace() {
    val middle by decisionSpace(::Middle)
}

private class AdSlot : SubSpace() {
    val premium by boolVar()
    val budget by intVar(0, 1000)
}

private class TwoAdSlots : DecisionSpace() {
    val slotA by decisionSpace(::AdSlot)
    val slotB by decisionSpace(::AdSlot)
}

private class Audio : SubSpace() {
    val mute by boolVar()
    val volume by intVar(0, 11)
}

private class OptionalAudio : DecisionSpace() {
    val audio by optionalDecisionSpace(::Audio)
}

private class OptionalAudioInner : SubSpace() {
    val inner by optionalDecisionSpace(::AudioLeaf)
}

private class AudioLeaf : SubSpace() {
    val volume by intVar(0, 11)
}

private class NestedOptionals : DecisionSpace() {
    val outer by optionalDecisionSpace(::OptionalAudioInner)
}

class SubSpaceTest {

    @Test
    fun `sub-space variables should get qualified klause names`() {
        val space = WithOneSubModel().compileSpace()
        val boolIds = space.compiled.boolVarIdByName
        val intIds  = space.compiled.intVarIdByName

        assertTrue("gate" in boolIds, "expected 'gate' at root; have ${boolIds.keys}")
        assertTrue("area.flag" in boolIds, "expected 'area.flag'; have ${boolIds.keys}")
        assertTrue("area.budget" in intIds, "expected 'area.budget' as int; have ${intIds.keys}")
    }

    @Test
    fun `decisionSpaceDef should reflect the hierarchy with local names`() {
        val space = WithOneSubModel().compileSpace()
        val def = space.definition
        // Root variables: just `gate`. The nested CodeFeature lives under spaces.
        assertEquals(setOf("gate"), def.variables.keys)
        assertEquals(setOf("area"), def.spaces.keys)
        val area = def.spaces.getValue("area")
        assertEquals(setOf("flag", "budget"), area.variables.keys) // local names, not "area.flag"
        assertTrue(area.variables["flag"] is BoolSpec)
        val budget = area.variables["budget"]
        assertTrue(budget is IntSpec)
        assertEquals(0, (budget as IntSpec).min)
        assertEquals(100, budget.max)
    }

    @Test
    fun `multiple instances of same sub-space should get distinct namespaces`() {
        val model = TwoAdSlots()
        val space = model.compileSpace()
        val boolIds = space.compiled.boolVarIdByName

        assertTrue("slotA.premium" in boolIds)
        assertTrue("slotB.premium" in boolIds)

        // Their handles point at different klause variables.
        assertEquals("slotA.premium", model.slotA.premium.name)
        assertEquals("slotB.premium", model.slotB.premium.name)
        assertTrue(model.slotA.premium !== model.slotB.premium)
    }

    @Test
    fun `nested sub-spaces should compose into a tree`() {
        val model = Nested()
        val space = model.compileSpace()
        val def = space.definition
        assertEquals(setOf("middle"), def.spaces.keys)
        val middle = def.spaces.getValue("middle")
        assertEquals(setOf("knob"), middle.variables.keys)
        assertEquals(setOf("inner"), middle.spaces.keys)
        val inner = middle.spaces.getValue("inner")
        assertEquals(setOf("toggle"), inner.variables.keys)

        // Handles still carry fully-qualified names for klause lookups.
        assertEquals("middle.inner.toggle", model.middle.inner.toggle.name)
        assertEquals("middle.knob", model.middle.knob.name)
    }

    @Test
    fun `optional sub-space should appear under optionalSpaces and pin children when off`() {
        val model = OptionalAudio()
        val space = model.compileSpace()
        val def = space.definition

        assertTrue("audio" in def.optionalSpaces.keys)
        val audio = def.optionalSpaces.getValue("audio")
        assertEquals(setOf("mute", "volume"), audio.variables.keys)

        // klause sees the gate variable + auto-pinning constraints.
        val gate = space.gateOf(model.audio) ?: error("optional sub-space must expose its gate")
        assertEquals("audio", gate.name)
        val solver = LocalSearchSolver(space.compiled.problem)
        repeat(50) { seed ->
            val sample = solver.sample(LocalSearchParams(randomSeed = seed.toLong()))!!
            val gateOn = space.compiled.decode(gate, sample)
            if (!gateOn) {
                assertEquals(false, space.compiled.decode(model.audio.mute, sample),
                    "audio.mute should be pinned to false when gate is off")
                assertEquals(0, space.compiled.decode(model.audio.volume, sample),
                    "audio.volume should be pinned to 0 when gate is off")
            }
            assertEquals(gateOn, space.isActive(model.audio.mute, sample))
        }
    }

    @Test
    fun `nested optionals should compose active conditions`() {
        val model = NestedOptionals()
        val space = model.compileSpace()
        val def = space.definition
        // Outer optional sub-space contains an optional sub-space "inner" → leaf "volume".
        assertTrue("outer" in def.optionalSpaces.keys)
        val outer = def.optionalSpaces.getValue("outer")
        assertTrue("inner" in outer.optionalSpaces.keys)
        val inner = outer.optionalSpaces.getValue("inner")
        assertEquals(setOf("volume"), inner.variables.keys)

        // Active condition for the leaf should require both gates.
        val cond = space.activeConditions["outer.inner.volume"]
        assertTrue(cond is com.eignex.klause.ast.And)
        cond as com.eignex.klause.ast.And
        assertEquals(2, cond.children.size)
    }

    @Test
    fun `sub-space handles should carry qualified names end-to-end`() {
        val model = WithOneSubModel()
        val space = model.compileSpace()

        assertEquals("gate", model.gate.name)
        assertEquals("area.flag", model.area.flag.name)
        assertEquals("area.budget", model.area.budget.name)

        val solver = LocalSearchSolver(space.compiled.problem)
        val sample = solver.sample(LocalSearchParams(randomSeed = 1L))
        assertNotNull(sample, "solver must produce a feasible sample over a flat-bool problem")
        assertEquals(2, sample.bools.size)
        assertEquals(1, sample.ints.size)
        assertTrue(sample.ints[0] in 0..100)

        space.compiled.decode(model.area.flag, sample)  // should not throw
        val b = space.compiled.decode(model.area.budget, sample)
        assertTrue(b in 0..100)
    }
}
