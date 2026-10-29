package com.eignex.combo.shared

import com.eignex.combo.decisions.DecisionSpace
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.schema.eq
import com.eignex.klause.schema.plus
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.Sample
import com.eignex.klause.solver.result.SampleResult
import com.eignex.klause.solver.result.TerminationReason
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class WitnessSearchTest {
    private class Toggle : DecisionSpace() {
        val value by boolVar()
    }

    private fun point(value: Boolean) = Sample(booleanArrayOf(value), longArrayOf())

    @Test
    fun `partial batches preserve witnesses and count fallback calls against the budget`() {
        val space = Toggle().compileSpace()
        var calls = 0
        val search = HybridDescentSession(
            space,
            { _, _, _ ->
                if (calls++ == 0) {
                    SampleResult.Found(point(false))
                } else {
                    SampleResult.Unknown(TerminationReason.BudgetExhausted)
                }
            },
            { _, _ ->
                calls++
                SampleResult.Unknown(TerminationReason.BudgetExhausted)
            },
        )
        val batch = search.witnesses(WitnessRequest(2, maxAttempts = 3), Random(1))
        assertEquals(listOf(point(false)), batch.witnesses)
        assertEquals(WitnessTermination.BudgetExhausted, batch.termination)
        assertEquals(3, calls)
    }

    @Test
    fun `duplicate exhaustion is distinct from proven infeasibility`() {
        val space = Toggle().compileSpace()
        for (proof in listOf(false, true)) {
            val search = HybridDescentSession(
                space,
                { _, _, _ -> if (proof) SampleResult.Infeasible() else SampleResult.Found(point(false)) },
                { _, _ -> error("no fallback expected") },
            )
            val batch = search.witnesses(WitnessRequest(2, exclude = listOf(point(false)), maxAttempts = 2), Random(1))
            assertTrue(batch.witnesses.isEmpty())
            assertEquals(
                if (proof) WitnessTermination.Infeasible else WitnessTermination.BudgetExhausted,
                batch.termination,
            )
        }
    }

    @Test
    fun `batch reuse respects scope exclusions and immutable snapshots`() {
        val space = Toggle().compileSpace()
        var calls = 0
        val search = HybridDescentSession(
            space,
            { _, _, _ -> SampleResult.Found(point(calls++ % 2 == 0)) },
            { _, _ -> error("no fallback expected") },
            candidateBatchSize = 2,
        )
        val first = search.witnesses(WitnessRequest(2, maxAttempts = 2), Random(1))
        assertEquals(WitnessTermination.CountReached, first.termination)
        val second = search.witnesses(
            WitnessRequest(1, Assumptions(bools = mapOf(0 to true)), exclude = listOf(point(false)), maxAttempts = 0),
            Random(2),
        )
        assertEquals(listOf(point(true)), second.witnesses)
        assertEquals(2, first.witnesses.size)
        assertEquals(2, calls)
    }

    @Test
    fun `batches continue refreshing a populated cache`() {
        val space = Toggle().compileSpace()
        var calls = 0
        val search = HybridDescentSession(
            space,
            { _, _, _ -> SampleResult.Found(point(calls++ % 2 == 0)) },
            { _, _ -> error("no fallback expected") },
            candidateBatchSize = 1,
        )
        search.witnesses(WitnessRequest(1, maxAttempts = 1), Random(1))
        search.witnesses(WitnessRequest(1, maxAttempts = 1), Random(2))
        assertEquals(2, calls)
    }

    @Test
    fun `invalid request sizes and scoped batch calls are rejected`() {
        assertFailsWith<IllegalArgumentException> { WitnessRequest(0) }
        assertFailsWith<IllegalArgumentException> { WitnessRequest(1, maxAttempts = -1) }
        val search = HybridDescentSession.klause(Toggle().compileSpace())
        search.push(Assumptions.None)
        assertFailsWith<IllegalStateException> { search.witnesses(WitnessRequest(1), Random(1)) }
    }
    private class Affine : DecisionSpace() {
        val x by floatVar(0.0, 1.0)
        val y by floatVar(0.0, 1.0)
        val sum by constraint { (x + y) eq 1.0 }
    }

    @Test
    fun `stalled random exploration samples exact affine interiors within its attempt budget`() {
        val space = Affine().compileSpace()
        fun endpoint(x: Double) = Sample(
            booleanArrayOf(),
            longArrayOf(),
            doubleArrayOf(x, 1.0 - x),
            exactReals = listOf(BigFraction.ofDouble(x)!!, BigFraction.ofDouble(1.0 - x)!!),
        )
        val ends = listOf(endpoint(0.0), endpoint(1.0))
        var calls = 0
        val search = HybridDescentSession(
            space,
            { _, _, _ ->
                calls++
                SampleResult.Found(ends.first())
            },
            { _, _ -> error("no backtracking expected") },
            candidateBatchSize = 1,
        )
        val batch = search.witnesses(WitnessRequest(1, anchors = ends, exclude = ends, maxAttempts = 2), Random(9))
        assertEquals(WitnessTermination.CountReached, batch.termination)
        assertEquals(1, batch.witnesses.map { it.assignmentKey() }.distinct().size)
        assertEquals(1, calls)
        assertTrue(batch.witnesses.all { WitnessValidator(space).valid(it) })
        assertTrue(batch.witnesses.all { it.reals[0] > 0.0 && it.reals[0] < 1.0 })
        assertTrue(
            search.witnesses(
                WitnessRequest(
                    1,
                    exclude = ends + batch.witnesses,
                    maxAttempts = 0,
                ),
                Random(1),
            ).witnesses.isEmpty(),
        )
    }

    @Test
    fun `refresh explores affine interiors even when cached endpoints already fill a request`() {
        val space = Affine().compileSpace()
        val ends = listOf(0.0, 1.0).map { x ->
            Sample(
                booleanArrayOf(),
                longArrayOf(),
                doubleArrayOf(x, 1.0 - x),
                exactReals = listOf(BigFraction.ofDouble(x)!!, BigFraction.ofDouble(1.0 - x)!!),
            )
        }
        val search = HybridDescentSession(
            space,
            { _, _, _ -> SampleResult.Found(ends.first()) },
            { _, _ -> error("no backtracking expected") },
            candidateBatchSize = 2,
        )
        search.beginRound { point -> point.reals[0] * (1.0 - point.reals[0]) }
        search.witnesses(WitnessRequest(1, anchors = ends, maxAttempts = 2), Random(0))
        val interior = requireNotNull(search.bestWitness(Assumptions.None, Random(0)))
        assertTrue(interior.reals[0] > 0.0 && interior.reals[0] < 1.0)
        assertTrue(WitnessValidator(space).valid(interior))
    }
}
