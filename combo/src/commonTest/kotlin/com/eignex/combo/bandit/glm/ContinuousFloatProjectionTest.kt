package com.eignex.combo.bandit.glm

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.decisions.DecisionSpaceDef
import com.eignex.combo.decisions.SubSpace
import com.eignex.combo.shared.LinearFeatureProjection
import com.eignex.combo.test.TimedTest
import com.eignex.klause.schema.le
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.Sample
import com.eignex.koblas.DenseVector
import com.eignex.skema.SchemaJson
import kotlinx.serialization.encodeToString
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull

class ContinuousFloatProjectionTest : TimedTest() {
    private class Space : DecisionSpace() {
        val temperature by floatVar(-5.0, 5.0)
        val count by intVar(0, 10)
    }

    private class OptionalFloat : SubSpace() {
        val value by floatVar(-5.0, 5.0)
        val cap by constraint { value le 2.0 }
    }

    private class OptionalSpace : DecisionSpace() {
        val optional by optionalDecisionSpace(::OptionalFloat)
    }

    @Test
    fun `optional float subspaces mask absent values and retain serialized constraints`() {
        val model = OptionalSpace()
        val space = model.compileSpace()
        val restored = SchemaJson.decodeFromString<DecisionSpaceDef>(SchemaJson.encodeToString(space.definition))
        assertEquals(space.definition, restored)
        val gate = assertNotNull(space.gateOf(model.optional))
        val gateId = space.compiled.boolVarIdByName.getValue(gate.name)
        val raw = Sample(
            BooleanArray(space.compiled.problem.numBoolVars),
            longArrayOf(),
            doubleArrayOf(-5.0),
            exactReals = listOf(BigFraction.ofDouble(-5.0)!!),
        )
        assertFalse(raw.boolValue(gateId))
        assertEquals(-5.0, space.compiled.decode(model.optional.value, raw))
        val projection = LinearFeatureProjection(space)
        val features = projection.encode(BanditSample(raw), Context.Empty)
        assertEquals(0.0, features[projection.layout.realStart])
    }

    @Test
    fun `continuous features and fractional objectives use the same real coordinate`() {
        val model = Space()
        val space = model.compileSpace()
        val projection = LinearFeatureProjection(space)
        val layout = projection.layout
        assertEquals(1, layout.numRealVars)
        assertEquals(1, layout.numIntVars)
        val realId = layout.realHandles.entries.single().key
        val weights = DoubleArray(layout.featureSize)
        weights[layout.realStart + realId] = 0.125
        weights[layout.intStart] = -0.25

        for (maximize in listOf(false, true)) {
            val objective = projection.toObjective(DenseVector.of(weights), 123.0, Context.Empty, maximize)
            val scale = abs(objective.realCoefficients[realId] / 0.125)
            for ((temperature, count, expected) in listOf(
                Triple(-5.0, 10L, -3.125),
                Triple(1.25, 3L, -0.59375),
                Triple(5.0, 0L, 0.625),
            )) {
                val raw = Sample(
                    booleanArrayOf(),
                    longArrayOf(count),
                    doubleArrayOf(temperature),
                    exactReals = listOf(BigFraction.ofDouble(temperature)!!),
                )
                val sample = BanditSample(raw)
                val features = projection.encode(sample, Context.Empty)
                assertEquals(temperature, sample.float(model.temperature, space))
                assertEquals(count, space.compiled.decode(model.count, raw))
                assertEquals(temperature, features[layout.realStart + realId])
                val cost = objective.intCoefficients.single() * count + objective.realCoefficients[realId] * temperature
                assertEquals(if (maximize) -expected else expected, cost / scale, 1e-12)
            }
        }
    }
}
