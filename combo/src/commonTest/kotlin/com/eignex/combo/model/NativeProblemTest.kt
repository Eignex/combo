package com.eignex.combo.model

import com.eignex.combo.bo.SearchTransform
import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.DecisionSpaceDef
import com.eignex.combo.decisions.conditionedProblem
import com.eignex.combo.decisions.context
import com.eignex.combo.decisions.minimizeNative
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.combo.shared.LinearFeatureProjection
import com.eignex.combo.shared.WitnessRequest
import com.eignex.combo.shared.assignmentKey
import com.eignex.klause.arithmetic.real.compareExactly
import com.eignex.klause.arithmetic.real.exactRealValue
import com.eignex.klause.formats.json.ExactRealJson
import com.eignex.klause.formats.json.ProblemJson
import com.eignex.klause.ir.SourceRealId
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.Sample
import com.eignex.klause.solver.result.QuadraticOptimizationResult
import com.eignex.klause.util.bigIntOf
import com.eignex.skema.SchemaJson
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlin.random.Random
import kotlin.test.*

class NativeProblemTest {
    @Test
    fun `exact algebraic and symbolic contexts preserve matches and assignment keys`() {
        val problem = ProblemJson.default.decodeFromString("""{"bools":1,"reals":[{}]}""")
        val space = CompiledDecisionSpace.fromProblem(problem)
        for (text in listOf(
            """{"root":{"coefficients":["-2","0","1"],"index":2}}""",
            """{"add":[{"pi":{}},{"exp":"1"}]}""",
        )) {
            val value = ExactRealJson.decodeFromString(text)
            val given = context { set(SourceRealId(0), value) }
            val sample = Sample(
                booleanArrayOf(false),
                longArrayOf(),
                doubleArrayOf(if (text.contains("root")) kotlin.math.sqrt(2.0) else kotlin.math.PI + kotlin.math.E),
                symbolicReals = listOf(value),
            )
            assertEquals(0, sample.exactRealValue(0).compareExactly(value))
            assertTrue(space.matches(sample, given))
            val features = SearchTransform(space).encode(BanditSample(sample), given).toDoubleArray()
            assertTrue(features.all { it.isFinite() })
            val encoded = ExactRealJson.encodeToString(sample.exactRealValue(0))
            val decoded = Sample(
                sample.bools,
                sample.ints,
                sample.reals,
                symbolicReals = listOf(ExactRealJson.decodeFromString(encoded)),
            )
            assertEquals(sample.assignmentKey(), decoded.assignmentKey())
            val search = HybridDescentSession.klause(space, candidateBatchSize = 1)
            assertTrue(
                search.witnesses(
                    WitnessRequest(
                        1,
                        context = given,
                        anchors = listOf(sample),
                        exclude = listOf(decoded),
                        maxAttempts = 0,
                    ),
                    Random(42),
                ).witnesses.isEmpty(),
            )
        }
    }

    @Test
    fun `exact context matching distinguishes equal Double approximations and setter replacement`() {
        val space = CompiledDecisionSpace.fromProblem(ProblemJson.default.decodeFromString("""{"reals":[{}]}"""))
        val value = ExactRealJson.decodeFromString("""{"exp":"1"}""")
        val other = ExactRealJson.decodeFromString("""{"exp":"100000000000000000001/100000000000000000000"}""")
        val sample = Sample(
            booleanArrayOf(),
            longArrayOf(),
            doubleArrayOf(kotlin.math.E),
            symbolicReals = listOf(value),
        )
        assertTrue(space.matches(sample, context { set(SourceRealId(0), value) }))
        assertFalse(space.matches(sample, context { set(SourceRealId(0), other) }))
        assertFalse(space.matches(sample, context { set(SourceRealId(0), kotlin.math.E) }))
        val replaced = context {
            set(SourceRealId(0), value)
            set(SourceRealId(0), 1.0)
        }
        assertTrue(space.matches(Sample(booleanArrayOf(), longArrayOf(), doubleArrayOf(1.0)), replaced))
    }

    @Test
    fun `canonical problem document survives model serialization without schema lowering`() {
        val problem = ProblemJson.default.decodeFromString(
            """
            {"bools":2,"ints":[{"lower":3000000000,"upper":3000000010,"values":[[3000000000,3000000002],[3000000009,3000000010]]}],
             "reals":[{}],"factors":[{"clause":{"literals":[0,2]}},
               {"table":{"vars":[0],"tuples":[[3000000000],[3000000010]]}}],
             "implied":[1],"symmetryBreaking":true,"settings":{"largeSpanThreshold":123456},
             "semantics":{"declared":{"format":"SMTLIB"},"effective":{"format":"SMTLIB","pinAbsentOptionalValues":false}}}
            """.trimIndent(),
        )
        val definition = ModelDefinition(problem = problem)
        val document = Json.parseToJsonElement(SchemaJson.encodeToString(definition)).jsonObject
        assertEquals(ProblemJson.default.encodeToJsonElement(problem), document["problem"])
        val restored = SchemaJson.decodeFromString<ModelDefinition>(document.toString()).problem!!
        assertEquals(
            ProblemJson.default.encodeToJsonElement(problem),
            ProblemJson.default.encodeToJsonElement(restored),
        )
        assertEquals(problem.factors.map { it::class }, restored.factors.map { it::class })
        assertEquals(123456L, restored.settings.largeSpanThreshold)
        assertTrue(restored.semantics.hasOverrides)
        val space = CompiledDecisionSpace.fromProblem(restored)
        assertSame(restored, space.problem)
        assertSame(restored, space.conditionedProblem(Context.Empty))
        assertEquals(3000000010L, space.problem.intBounds.upper(0))
        assertFalse(3000000005L in space.problem.intDomainOrNull(0)!!)
        assertEquals(Double.NEGATIVE_INFINITY, space.problem.realLower[0])
        assertEquals(4, SearchTransform(space).featureSize)
        assertEquals(4, LinearFeatureProjection(space).featureSize)
        assertFailsWith<IllegalArgumentException> {
            ModelDefinition(problem = problem, space = DecisionSpaceDef())
        }
    }

    @Test
    fun `native solver respects sparse Long domains and real rational pins`() {
        val problem = ProblemJson.default.decodeFromString(
            """
            {"ints":[{"lower":3000000000,"upper":3000000010,"values":[[3000000000,3000000000],[3000000010,3000000010]]}],
             "reals":[{}],"factors":[{"linear":{"vars":[0],"coefficients":[1],"op":"GE","bound":3000000005}}],"implied":[0]}
            """.trimIndent(),
        )
        val space = CompiledDecisionSpace.fromProblem(problem)
        val third = BigFraction.of(bigIntOf(1), bigIntOf(3))
        val given = context { set(SourceRealId(0), third) }
        val result = assertIs<QuadraticOptimizationResult.Optimal>(
            space.minimizeNative(space.coordinates.objective("ints.0"), given),
        )
        assertEquals(3000000010L, result.incumbent.sample.intValue(0))
        val sample = result.incumbent.sample
        val rational = sample.symbolicReals?.get(0)?.rational ?: sample.algebraicReals?.get(0)?.rational
            ?: sample.exactReals?.get(0)
        assertEquals(third, rational)
        assertTrue(space.matches(result.incumbent.sample, given))
        val pinned = space.conditionedProblem(given)
        assertTrue(pinned.impliedFactorMask!![0])
        assertFalse(pinned.impliedFactorMask!![1])
        assertSame(problem.settings, pinned.settings)
        assertSame(problem.semantics, pinned.semantics)
    }

    @Test
    fun `native witness search feeds existing history and linear projections`() {
        val problem = ProblemJson.default.decodeFromString("""{"bools":1,"ints":[{"lower":0,"upper":3}]}""")
        val space = CompiledDecisionSpace.fromProblem(problem)
        val batch = HybridDescentSession.klause(
            space,
            candidateBatchSize = 1,
        ).witnesses(WitnessRequest(1, maxAttempts = 1), Random(42))
        assertEquals(1, batch.witnesses.size)
        for (sample in batch.witnesses) {
            assertEquals(2, SearchTransform(space).encode(BanditSample(sample), Context.Empty).size)
            assertEquals(2, LinearFeatureProjection(space).encode(BanditSample(sample), Context.Empty).size)
        }
    }

    @Test
    fun `native history transfer checks the problem rather than empty schema metadata`() {
        val codec = ProblemJson.default
        val problem = codec.decodeFromString("""{"ints":[{"lower":0,"upper":3}]}""")
        val snapshot = com.eignex.combo.bo.Study(CompiledDecisionSpace.fromProblem(problem)).exportData()
        val same = codec.decodeFromString(codec.encodeToString(problem))
        com.eignex.combo.bo.Study(CompiledDecisionSpace.fromProblem(same)).importData(snapshot)
        val changed = codec.decodeFromString("""{"ints":[{"lower":0,"upper":4}]}""")
        assertFailsWith<IllegalArgumentException> {
            com.eignex.combo.bo.Study(CompiledDecisionSpace.fromProblem(changed)).importData(snapshot)
        }
    }

    @Test
    fun `native integer feature geometry preserves adjacent large Long values`() {
        val problem = ProblemJson.default.decodeFromString(
            """
            {"ints":[{"lower":9007199254740992,"upper":9007199254740993}]}
            """.trimIndent(),
        )
        val space = CompiledDecisionSpace.fromProblem(problem)
        val transform = SearchTransform(space)
        val lower = BanditSample(com.eignex.klause.solver.Sample(booleanArrayOf(), longArrayOf(9007199254740992L)))
        val upper = BanditSample(com.eignex.klause.solver.Sample(booleanArrayOf(), longArrayOf(9007199254740993L)))
        assertEquals(0.0, transform.encode(lower, Context.Empty)[0])
        assertEquals(1.0, transform.encode(upper, Context.Empty)[0])
        val split = com.eignex.combo.bandit.dt.defaultSplitCandidates(space).single()
        val rows = com.eignex.combo.bandit.dt.TreeFeatureProjection(space)
        assertTrue(split.direction(rows.encode(lower, Context.Empty)))
        assertFalse(split.direction(rows.encode(upper, Context.Empty)))
        val layout = com.eignex.combo.bandit.dt.TreeVectorLayout(listOf(split))
        assertEquals(0.0, layout.encode(rows.encode(lower, Context.Empty))[0])
        assertEquals(1.0, layout.encode(rows.encode(upper, Context.Empty))[0])
    }

    @Test
    fun `rational contexts do not match a different real with the same Double approximation`() {
        val problem = ProblemJson.default.decodeFromString("""{"reals":[{}]}""")
        val space = CompiledDecisionSpace.fromProblem(problem)
        val third = BigFraction.of(bigIntOf(1), bigIntOf(3))
        val exact = com.eignex.klause.solver.Sample(
            booleanArrayOf(),
            longArrayOf(),
            doubleArrayOf(third.toDouble()),
            exactReals = listOf(third),
        )
        assertTrue(space.matches(exact, context { set(SourceRealId(0), third) }))
        assertFalse(space.matches(exact, context { set(SourceRealId(0), third.toDouble()) }))
    }
}
