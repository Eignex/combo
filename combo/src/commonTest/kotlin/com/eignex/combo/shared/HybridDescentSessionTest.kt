package com.eignex.combo.shared

import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.test.TimedTest
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.solver.Sample
import com.eignex.klause.solver.result.SampleResult
import com.eignex.klause.solver.result.TerminationReason
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

class HybridDescentSessionTest : TimedTest() {
    private class Space : DecisionSpace() {
        val a by boolVar()
        val b by boolVar()
        val count by intVar(0, 10)
    }

    private fun point(a: Boolean, b: Boolean, count: Long = 0L) = Sample(booleanArrayOf(a, b), longArrayOf(count))

    @Test
    fun `compatible branches reuse witnesses without solving again`() {
        val space = Space().compileSpace()
        var calls = 0
        val samples = listOf(point(false, false), point(true, false), point(true, true))
        val session = HybridDescentSession(
            space,
            { _, _, _ -> SampleResult.Found(samples[calls++ % samples.size]) },
            { _, _ -> error("backtracking should not run") },
            candidateBatchSize = 3,
        )
        session.beginRound { if (it.boolValue(1)) 1.0 else 0.0 }
        session.push(Assumptions.None)
        assertSame(samples[2], session.sample(Random(1)))
        session.push(Assumptions(bools = mapOf(0 to true)))
        assertSame(samples[2], session.sample(Random(2)))
        assertEquals(3, calls)
        assertEquals(2L, session.poolHits)
        assertEquals(0L, session.backtrackCalls)
        session.pop()
        session.pop()
        assertEquals(0, session.depth)
    }

    @Test
    fun `a missing branch warm starts repair then backtracks without losing the incumbent`() {
        val space = Space().compileSpace()
        val original = point(false, false)
        val repaired = point(true, false)
        val session = HybridDescentSession(
            space,
            propose = { _, assumptions, anchor ->
                if (assumptions.isEmpty) {
                    SampleResult.Found(original)
                } else {
                    assertSame(original, anchor)
                    SampleResult.Unknown(TerminationReason.BudgetExhausted)
                }
            },
            backtrack = { _, assumptions ->
                assertEquals(true, assumptions.boolValueOrNull(0))
                SampleResult.Found(repaired)
            },
            candidateBatchSize = 1,
        )
        session.beginRound { if (it.boolValue(0)) 1.0 else 0.0 }
        session.push(Assumptions.None)
        assertSame(original, session.sample(Random(1)))
        session.push(Assumptions(bools = mapOf(0 to true)))
        assertSame(repaired, session.sample(Random(2)))
        assertEquals(1L, session.backtrackCalls)
        session.pop()
        assertSame(repaired, session.bestWitness(Assumptions.None, Random(3)))
        session.pop()
    }

    @Test
    fun `unknown and proven infeasible remain distinct outcomes`() {
        val space = Space().compileSpace()
        for (outcome in listOf(
            SampleResult.Unknown(TerminationReason.BudgetExhausted),
            SampleResult.Infeasible(),
        )) {
            val session = HybridDescentSession(
                space,
                { _, _, _ -> SampleResult.Unknown(TerminationReason.BudgetExhausted) },
                { _, _ -> outcome },
                candidateBatchSize = 1,
            )
            assertNull(session.sample(Random(1)))
            assertSame(outcome, session.lastResult)
            assertEquals(1L, session.backtrackCalls)
        }
    }

    @Test
    fun `cached witnesses must satisfy integer bounds holes and fixed context`() {
        val space = Space().compileSpace()
        val candidates = listOf(point(false, false, 2), point(true, false, 9))
        var calls = 0
        val session = HybridDescentSession(
            space,
            { _, _, _ -> SampleResult.Found(candidates[calls++ % 2]) },
            { _, _ -> error("backtracking should not run") },
            candidateBatchSize = 2,
        )
        session.beginRound { 0.0 }
        session.push(Assumptions.None)
        assertNotNull(session.sample(Random(1)))
        val scope = Assumptions(bools = mapOf(0 to true)).withTightenedMin(0, 5).withIntHole(0, 8)
        session.push(scope)
        val sample = assertNotNull(session.sample(Random(2)))
        assertTrue(sample.boolValue(0))
        assertEquals(9L, sample.intValue(0))
        assertIs<SampleResult.Found>(session.lastResult)
    }
}
