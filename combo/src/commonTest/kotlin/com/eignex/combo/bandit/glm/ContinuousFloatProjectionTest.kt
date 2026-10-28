package com.eignex.combo.bandit.glm

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.decisions.SubSpace
import com.eignex.combo.shared.LinearFeatureProjection
import com.eignex.combo.test.TimedTest
import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.backtrack.BacktrackSolver
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.propagation.bake
import com.eignex.klause.schema.le
import com.eignex.koblas.DenseVector
import kotlin.test.Test
import kotlin.test.assertEquals
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
    fun `optional float subspaces pin absent values and round-trip their constraints`() {
        val model = OptionalSpace()
        val space = model.compileSpace()
        val gate = assertNotNull(space.gateOf(model.optional))
        val gateId = space.compiled.boolVarIdByName.getValue(gate.name)
        val assumptions = Assumptions(bools = mapOf(gateId to false))
        val projection = LinearFeatureProjection(space)
        for (compiled in listOf(space.compiled, space.definition.compile())) {
            val raw = assertNotNull(
                BacktrackSolver(compiled.problem.bake()).sample(
                    BacktrackParams(assumptions = assumptions),
                ).assignment,
            )
            assertEquals(-5.0, compiled.decode(model.optional.value, raw), 1e-8)
        }
        val raw = assertNotNull(
            BacktrackSolver(space.compiled.problem.bake()).sample(
                BacktrackParams(assumptions = assumptions),
            ).assignment,
        )
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
        val solver = BacktrackSolver(space.compiled.problem.bake())

        for (maximize in listOf(false, true)) {
            val objective = projection.toObjective(DenseVector.of(weights), 123.0, Context.Empty, maximize)
            val raw = assertNotNull(solver.minimize(objective, BacktrackParams()).assignment)
            val sample = BanditSample(raw)
            val features = projection.encode(sample, Context.Empty)
            assertEquals(if (maximize) 5.0 else -5.0, sample.float(model.temperature, space), 1e-8)
            assertEquals(if (maximize) 0L else 10L, space.compiled.decode(model.count, raw))
            assertEquals(space.compiled.decode(model.temperature, raw), features[layout.realStart + realId], 1e-8)
        }
    }
}
