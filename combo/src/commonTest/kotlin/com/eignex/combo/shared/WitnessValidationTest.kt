package com.eignex.combo.shared

import com.eignex.combo.bo.ProposalBatch
import com.eignex.combo.bo.Study
import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.decisions.context
import com.eignex.klause.formats.json.ExactRealJson
import com.eignex.klause.formats.json.ProblemJson
import com.eignex.klause.ir.SourceRealId
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.schema.eq
import com.eignex.klause.schema.plus
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.Sample
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class WitnessValidationTest {
    private class Affine : DecisionSpace() {
        val x by floatVar(0.0, 1.0)
        val y by floatVar(0.0, 1.0)
        val sum by constraint { (x + y) eq 1.0 }
    }

    @Test
    fun `source validation rejects an exact equality violation hidden by Double rounding`() {
        val space = Affine().compileSpace()
        val half = BigFraction.ofDouble(0.5)!!
        val different = half + BigFraction.ofDouble(1e-20)!!
        fun point(y: BigFraction) = Sample(
            booleanArrayOf(),
            longArrayOf(),
            doubleArrayOf(0.5, y.toDouble()),
            exactReals = listOf(half, y),
        )
        val valid = point(half)
        val invalid = point(different)
        assertEquals(valid.reals.toList(), invalid.reals.toList())
        val validator = WitnessValidator(space)
        assertTrue(validator.valid(valid))
        assertFalse(validator.valid(invalid))
        val search = HybridDescentSession(
            space,
            { _, _, _ -> error("zero attempt budget must not propose") },
            { _, _ -> error("zero attempt budget must not backtrack") },
        )
        val batch = search.witnesses(WitnessRequest(1, anchors = listOf(invalid), maxAttempts = 0), Random(1))
        assertTrue(batch.witnesses.isEmpty())
        assertEquals(WitnessTermination.BudgetExhausted, batch.termination)
        val study = Study(space)
        val snapshot = study.beginRound()
        assertFailsWith<IllegalArgumentException> {
            study.reserve(
                snapshot,
                ProposalBatch(listOf(BanditSample(valid), BanditSample(invalid)), WitnessTermination.CountReached),
            )
        }
        assertTrue(study.trials.isEmpty())
        val reserved = study.reserve(
            snapshot,
            ProposalBatch(listOf(BanditSample(valid)), WitnessTermination.CountReached),
        )
        assertEquals(1L, reserved[0].id)
    }

    @Test
    fun `native exact JSON identity controls context matching and pending reservations`() {
        val space = CompiledDecisionSpace.fromProblem(ProblemJson.default.decodeFromString("""{"reals":[{}]}"""))
        for (text in listOf(
            """{"root":{"coefficients":["-2","0","1"],"index":2}}""",
            """{"exp":"1"}""",
        )) {
            val exact = ExactRealJson.decodeFromString(text)
            val restored = ExactRealJson.decodeFromString(ExactRealJson.encodeToString(exact))
            val point = Sample(booleanArrayOf(), longArrayOf(), doubleArrayOf(1.0), symbolicReals = listOf(exact))
            val replay = Sample(booleanArrayOf(), longArrayOf(), doubleArrayOf(2.0), symbolicReals = listOf(restored))
            assertEquals(point.assignmentKey(), replay.assignmentKey())
            val given = context { set(SourceRealId(0), restored) }
            val study = Study(space)
            val first = study.reserve(
                study.beginRound(),
                ProposalBatch(listOf(BanditSample(point)), WitnessTermination.CountReached),
                given,
            )
            val snapshot = study.beginRound()
            assertFailsWith<IllegalArgumentException> {
                study.reserve(
                    snapshot,
                    ProposalBatch(listOf(BanditSample(replay)), WitnessTermination.CountReached),
                    given,
                )
            }
            assertEquals(1, study.trials.size)
            study.cancel(first[0].id)
            val next = study.reserve(
                study.beginRound(),
                ProposalBatch(listOf(BanditSample(replay)), WitnessTermination.CountReached),
                given,
            )
            assertEquals(2L, next[0].id)
        }
    }

    @Test
    fun `real context requests cannot bypass the empty assumption stack requirement`() {
        val space = Affine().compileSpace()
        val session = HybridDescentSession(
            space,
            { _, _, _ -> error("scoped batch must not propose") },
            { _, _ -> error("scoped batch must not backtrack") },
            conditionedSearch = { throw AssertionError("scoped batch must not delegate") },
        )
        session.push(Assumptions.None)
        val given = context { set(SourceRealId(0), 0.5) }
        assertFailsWith<IllegalStateException> {
            session.witnesses(WitnessRequest(1, context = given), Random(1))
        }
        assertEquals(1, session.depth)
        session.pop()
    }
}
