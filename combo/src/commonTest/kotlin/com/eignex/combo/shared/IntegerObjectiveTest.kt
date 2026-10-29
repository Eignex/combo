package com.eignex.combo.shared

import com.eignex.combo.decisions.DecisionSpace
import com.eignex.klause.solver.Sample
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class IntegerObjectiveTest {
    private class Space : DecisionSpace() {
        val a by boolVar()
        val b by boolVar()
        val count by intVar(Int.MIN_VALUE, Int.MAX_VALUE)
    }

    @Test
    fun `fractional scores retain their ordering at tiny and huge magnitudes`() {
        val problem = Space().compileSpace().compiled.problem
        for (magnitude in listOf(1e-320, 1e-200, 1.0, 1e200)) {
            val objective = integerObjective(problem, doubleArrayOf(0.25 * magnitude, -0.75 * magnitude))
            val costs = (0..3).map { bits ->
                objective.evaluateLong(Sample(booleanArrayOf(bits and 1 != 0, bits and 2 != 0), longArrayOf(0)))
            }
            assertTrue(costs[2] < costs[3] && costs[3] < costs[0] && costs[0] < costs[1])
            assertTrue(kotlin.math.abs(3L * objective.boolWeights[0] + objective.boolWeights[1]) <= 1L)
        }
    }

    @Test
    fun `wide integer domains keep objective arithmetic within its budget`() {
        val problem = Space().compileSpace().compiled.problem
        val objective = integerObjective(problem, doubleArrayOf(0.25, -0.75), doubleArrayOf(0.1))
        for (count in listOf(Int.MIN_VALUE.toLong(), Int.MAX_VALUE.toLong())) {
            val cost = objective.evaluateLong(Sample(booleanArrayOf(true, true), longArrayOf(count)))
            assertTrue(kotlin.math.abs(cost) < (1L shl 50))
        }
        assertTrue(objective.intCoefficients.single() > 0)
    }

    @Test
    fun `non-finite scores fail before entering the solver`() {
        val problem = Space().compileSpace().compiled.problem
        for (invalid in listOf(Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY)) {
            assertFailsWith<IllegalArgumentException> {
                integerObjective(problem, doubleArrayOf(invalid, 0.0))
            }
        }
    }
}
