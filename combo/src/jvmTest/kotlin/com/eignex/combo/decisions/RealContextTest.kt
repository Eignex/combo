package com.eignex.combo.decisions

import com.eignex.combo.model.LinearLearning
import com.eignex.combo.shared.LinearFeatureProjection
import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.backtrack.BacktrackSolver
import com.eignex.klause.model.FloatLinearConstraint
import com.eignex.klause.model.FloatSpec
import com.eignex.klause.model.IntCmpOp
import com.eignex.klause.propagation.bake
import com.eignex.klause.solver.SolveResult
import com.eignex.koblas.DenseVector
import com.eignex.kumulant.schema.spec.BayesianRegression
import com.eignex.kumulant.stat.regression.glm.MultivariateGaussian
import com.eignex.skema.SchemaJson
import kotlinx.serialization.encodeToString
import kotlin.test.*

private class RealContextSpace(private val grid: Int? = null) : DecisionSpace() {
    val capacity by realVar(0.0, 10.0)
    val load by contextFloat(0.0, 10.0, buckets = grid)
    val interaction by interact(load, capacity)
    val enough by constraint {
        FloatLinearConstraint(doubleArrayOf(1.0, -1.0), listOf("capacity", "load"), IntCmpOp.GE, 0.0)
    }
}

class RealContextTest {
    @Test
    fun `real contexts remain pinned across changing requests and fold into native linear objectives`() {
        val space = RealContextSpace().compileSpace()
        val projection = LinearFeatureProjection(space)
        val coefficients = DoubleArray(projection.featureSize)
        val capacity = projection.layout.realHandles.entries.single { it.value.name == "capacity" }.key
        coefficients[projection.layout.realStart + capacity] = 1.0
        val learning = LinearLearning(
            BayesianRegression(projection.featureSize, priorMean = DenseVector.of(coefficients)),
            MultivariateGaussian,
        )
        BacktrackSolver(space.compiled.problem.bake()).session().use { session ->
            val bandit = learning.materialize(
                projection,
                { objective, pins -> session.minimize(objective, BacktrackParams(assumptions = pins)).assignment },
                seed = 42, maximize = false,
            )
            val handle = space.contextFloats.single()
            for (value in listOf(2.5, 7.25, 1.125)) {
                val given = context { set(handle, value) }
                val result = bandit.optimalOrThrow(given)
                assertTrue(space.matches(result.sample, given))
                assertEquals(value, space.compiled.decode(handle.klauseHandle, result.sample))
                val actual = space.compiled.decode(
                    projection.layout.realHandles.getValue(capacity), result.sample,
                )
                assertEquals(value, actual, 1e-10)
                assertFailsWith<IllegalArgumentException> { space.assumptionsFor(given) }
            }
        }
        val definition = SchemaJson.decodeFromString<DecisionSpaceDef>(SchemaJson.encodeToString(space.definition))
        assertEquals(FloatSpec(0.0, 10.0), definition.context.getValue("load"))
        assertEquals(ScalarKind.FloatContext, definition.interactions.single().lhs.kind)
        assertEquals(ScalarKind.FloatDecision, definition.interactions.single().rhs.kind)
        assertEquals(1, definition.compileSpace().contextFloats.size)
    }

    @Test
    fun `float context follows the native representation without rounding`() {
        val space = RealContextSpace(grid = 5).compileSpace()
        val handle = space.contextFloats.single()
        val given = context { set(handle, 2.5) }
        val result = BacktrackSolver(space.conditionedProblem(given).bake()).solve(
            BacktrackParams(assumptions = space.discreteAssumptionsFor(given)),
        )
        assertIs<SolveResult.Sat>(result)
        assertTrue(space.matches(result.assignment.unframed(), given))
        val fractional = context { set(handle, 2.25) }
        if (space.compiled.minimize(handle.klauseHandle).realCoefficients.any { it != 0.0 }) {
            val native = BacktrackSolver(space.conditionedProblem(fractional).bake()).solve(
                BacktrackParams(assumptions = space.discreteAssumptionsFor(fractional)),
            )
            assertIs<SolveResult.Sat>(native)
            assertEquals(2.25, space.compiled.decode(handle.klauseHandle, native.assignment.unframed()))
        } else {
            assertFailsWith<IllegalArgumentException> { space.discreteAssumptionsFor(fractional) }
        }
    }
}
