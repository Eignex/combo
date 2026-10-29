package com.eignex.combo.decisions

import com.eignex.combo.bandit.dt.FloatThresholdSplit
import com.eignex.combo.bandit.dt.defaultSplitCandidates
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.combo.shared.LinearFeatureProjection
import com.eignex.combo.shared.WitnessRequest
import com.eignex.klause.model.IntSpec
import com.eignex.klause.solver.Sample
import com.eignex.koblas.DenseVector
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlin.math.ln
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ScaleTest {
    private class Child : SubSpace() {
        val rate by floatVar(1e-6, 1.0, scale = Scale.Log)
    }

    private class Scaled : DecisionSpace() {
        val rate by floatVar(1e-6, 1.0, scale = Scale.Log)
        val size by intVar(1, 100, scale = Scale.Log)
        val child by optionalDecisionSpace(::Child)
        val segment by contextInt(1, 100, scale = Scale.Log)
        val flag by boolVar()
        val effect by interact(segment, flag)
    }

    @Test
    fun `scale metadata retains local names through nested definition serialization`() {
        val space = Scaled().compileSpace()
        val definition = space.definition
        assertEquals(mapOf("rate" to Scale.Log, "size" to Scale.Log, "segment" to Scale.Log), definition.scales)
        assertEquals(mapOf("rate" to Scale.Log), definition.optionalSpaces.getValue("child").scales)
        val decoded = Json.decodeFromString<DecisionSpaceDef>(Json.encodeToString(definition))
        assertEquals(definition, decoded)
        assertEquals(space.compiled.problem.numRealVars, decoded.compile().problem.numRealVars)
        assertTrue("child.rate" in space.numericDomains)
    }

    @Test
    fun `numeric geometry is invertible at endpoints and logarithmic midpoints`() {
        val log = NumericDomain(1e-6, 1.0, Scale.Log)
        assertEquals(1e-6, log.fromUnit(0.0))
        assertEquals(1.0, log.fromUnit(1.0))
        assertEquals(1e-3, log.fromUnit(0.5), 1e-15)
        assertEquals(3.0, NumericDomain(-2.0, 8.0, Scale.Linear).fromUnit(0.5))
        val reverse = NumericDomain(0.0, 99.0, Scale.ReverseLog)
        assertEquals(90.0, reverse.fromUnit(0.5), 1e-12)
        assertEquals(0.0, reverse.fromUnit(0.0))
        assertEquals(99.0, reverse.fromUnit(1.0))
    }

    @Test
    fun `logarithmic bounds and metadata names are validated`() {
        assertFailsWith<IllegalArgumentException> {
            DecisionSpaceDef(variables = mapOf("x" to IntSpec(0, 100)), scales = mapOf("x" to Scale.Log))
        }
        assertFailsWith<IllegalArgumentException> { DecisionSpaceDef(scales = mapOf("missing" to Scale.Log)) }
    }

    private class Interaction : DecisionSpace() {
        val size by intVar(1, 100, scale = Scale.Log)
        val segment by contextInt(1, 100, scale = Scale.Log)
        val effect by interact(segment, size)
    }

    @Test
    fun `numeric features and interactions share transforms and reject incorrect linear objectives`() {
        val model = Interaction()
        val space = model.compileSpace()
        val projection = LinearFeatureProjection(space)
        val ints = LongArray(space.compiled.problem.numIntVars) { 10L }
        val sample = BanditSample(Sample(booleanArrayOf(), ints))
        val ctx = context { set(model.segment, 10) }
        val features = projection.encode(sample, ctx)
        assertEquals(ln(10.0), features[0], 1e-12)
        assertEquals(ln(10.0), features[1], 1e-12)
        assertEquals(ln(10.0) * ln(10.0), features[2], 1e-12)
        assertFailsWith<IllegalArgumentException> {
            projection.toObjective(DenseVector.zero(projection.featureSize), 0.0, ctx, true)
        }
    }

    private class Rate : DecisionSpace() {
        val rate by floatVar(1e-6, 1.0, scale = Scale.Log)
    }

    @Test
    fun `fresh real proposals include independently checked interiors across orders of magnitude`() {
        val model = Rate()
        val space = model.compileSpace()
        val search = HybridDescentSession.klause(space, candidateBatchSize = 1)
        val batch = search.witnesses(WitnessRequest(4, maxAttempts = 4), Random(31))
        val values = batch.witnesses.map { BanditSample(it).float(model.rate, space) }
        assertEquals(4, values.distinct().size)
        assertTrue(values.all { it in 1e-6..1.0 })
        assertTrue(values.any { it < 1e-3 })
        assertTrue(values.any { it in 1e-5..1e-2 })
        assertTrue(batch.witnesses.all { it.exactReals != null })
        val threshold = defaultSplitCandidates(
            space,
            numericThresholds = 1,
        ).filterIsInstance<FloatThresholdSplit>().single()
        assertEquals(1e-3, threshold.threshold, 1e-15)
    }
}
