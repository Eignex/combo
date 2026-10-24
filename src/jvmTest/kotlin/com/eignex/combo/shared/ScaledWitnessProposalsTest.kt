package com.eignex.combo.shared

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.decisions.Scale
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.schema.eq
import com.eignex.klause.schema.le
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.Sample
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ScaledWitnessProposalsTest {
    private class Capped : DecisionSpace() {
        val amount by floatVar(1e-6, 1.0, scale = Scale.Log)
        val cap by constraint { amount le 0.01 }
        val premium by contextBool()
    }

    @Test
    fun `proposals retain source feasibility and hard context under a narrower affine constraint`() {
        val model = Capped()
        val space = model.compileSpace()
        val id = space.compiled.boolVarIdByName.getValue(model.premium.name)
        val search = HybridDescentSession.klause(space)
        val batch = search.witnesses(WitnessRequest(20, Assumptions(bools = mapOf(id to true))), Random(4))
        assertEquals(20, batch.witnesses.size)
        assertTrue(batch.witnesses.all { it.boolValue(id) })
        assertTrue(batch.witnesses.all { BanditSample(it).float(model.amount, space) in 1e-6..0.01 })
        val validator = WitnessValidator(space)
        assertTrue(batch.witnesses.all(validator::valid))
    }

    @Test
    fun `unverified anchors are checked against constraints before reuse`() {
        val model = Capped()
        val space = model.compileSpace()
        fun point(value: Double) = Sample(
            booleanArrayOf(true),
            longArrayOf(),
            doubleArrayOf(value),
            exactReals = listOf(BigFraction.ofDouble(value)!!),
        )
        val valid = point(0.001)
        val invalid = point(0.5)
        val search = HybridDescentSession.klause(space)
        val batch = search.witnesses(WitnessRequest(2, anchors = listOf(valid, invalid), maxAttempts = 0), Random(1))
        assertEquals(listOf(valid), batch.witnesses)
        assertEquals(WitnessTermination.BudgetExhausted, batch.termination)
    }

    @Test
    fun `assignment exclusion ignores proof metadata`() {
        val model = Capped()
        val space = model.compileSpace()
        val exact = Sample(
            booleanArrayOf(true),
            longArrayOf(),
            doubleArrayOf(0.001),
            exactReals = listOf(BigFraction.ofDouble(0.001)!!),
        )
        val approximate = Sample(booleanArrayOf(true), longArrayOf(), doubleArrayOf(0.001))
        val search = HybridDescentSession.klause(space)
        val batch = search.witnesses(
            WitnessRequest(1, anchors = listOf(exact), exclude = listOf(approximate), maxAttempts = 0),
            Random(1),
        )
        assertTrue(batch.witnesses.isEmpty())
    }

    private class ZeroBound : DecisionSpace() {
        val amount by floatVar(0.0, 1.0)
    }

    @Test
    fun `signed zero cannot bypass a reserved assignment`() {
        val space = ZeroBound().compileSpace()
        val anchor = Sample(
            booleanArrayOf(),
            longArrayOf(),
            doubleArrayOf(0.0),
            exactReals = listOf(BigFraction.ZERO),
        )
        val reserved = Sample(booleanArrayOf(), longArrayOf(), doubleArrayOf(-0.0))
        val batch = HybridDescentSession.klause(space).witnesses(
            WitnessRequest(1, anchors = listOf(anchor), exclude = listOf(reserved), maxAttempts = 0),
            Random(1),
        )
        assertTrue(batch.witnesses.isEmpty())
    }

    private class FixedInteger : DecisionSpace() {
        val size by intVar(1, 1000, scale = Scale.Log)
        val amount by floatVar(0.0, 1.0)
        val fixed by constraint { size eq 10 }
    }

    @Test
    fun `failed exploratory integer pins do not prove the original scope infeasible`() {
        val model = FixedInteger()
        val space = model.compileSpace()
        val id = space.compiled.intVarIdByName.getValue(model.size.name)
        val search = HybridDescentSession.klause(space)
        val batch = search.witnesses(WitnessRequest(8), Random(17))
        assertEquals(WitnessTermination.CountReached, batch.termination)
        assertEquals(8, batch.witnesses.size)
        assertTrue(batch.witnesses.all { it.intValue(id) == 10L })
    }

    private class Mixed : DecisionSpace() {
        val size by intVar(1, 1000, scale = Scale.Log)
        val amount by floatVar(0.0, 1.0)
    }

    private class MixedLinear : DecisionSpace() {
        val count by intVar(1, 6)
        val mode by nominal("slow", "fast")
        val finite by floatVar(-2.0, 2.0, buckets = 9)
        val amount by floatVar(0.0, 1.0)
        val absent by optionalContextInt(1, 100)
    }

    @Test
    fun `real witness initialization explores linear integers categories and finite floats`() {
        val model = MixedLinear()
        val space = model.compileSpace()
        val context = space.assumptionsFor(com.eignex.combo.decisions.Context.Empty)
        val countId = space.compiled.intVarIdByName.getValue(model.count.name)
        val search = HybridDescentSession.klause(space)
        val batch = search.witnesses(WitnessRequest(64, context), Random(23))
        assertEquals(64, batch.witnesses.size)
        assertEquals((1L..6L).toSet(), batch.witnesses.map { it.intValue(countId) }.toSet())
        assertEquals(setOf("slow", "fast"), batch.witnesses.map { space.compiled.decode(model.mode, it) }.toSet())
        assertTrue(batch.witnesses.map { BanditSample(it).float(model.finite, space) }.distinct().size > 2)
        assertTrue(batch.witnesses.all { space.decode(model.absent, it) == 1 && !space.isActive(model.absent, it) })
        assertTrue(batch.witnesses.all(WitnessValidator(space)::valid))
        val pinned = search.witnesses(
            WitnessRequest(8, context.mergedWith(Assumptions(ints = mapOf(countId to 4L)))),
            Random(31),
        )
        assertEquals(8, pinned.witnesses.size)
        assertTrue(pinned.witnesses.all { it.intValue(countId) == 4L })
    }

    @Test
    fun `mixed real and integer proposals honor logarithmic geometry without replacing hard pins`() {
        val model = Mixed()
        val space = model.compileSpace()
        val id = space.compiled.intVarIdByName.getValue(model.size.name)
        val search = HybridDescentSession.klause(space)
        val varied = search.witnesses(WitnessRequest(40), Random(7))
        assertTrue(varied.witnesses.map { it.intValue(id) }.distinct().size > 10)
        assertTrue(varied.witnesses.count { it.intValue(id) <= 32L } > 8)
        val fixed = search.witnesses(WitnessRequest(10, Assumptions(ints = mapOf(id to 900L))), Random(9))
        assertEquals(10, fixed.witnesses.size)
        assertTrue(fixed.witnesses.all { it.intValue(id) == 900L })
    }
}
