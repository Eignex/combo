package com.eignex.combo.model

import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.DecisionSpaceDef
import com.eignex.combo.decisions.BanditSample
import kotlinx.serialization.json.Json
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.context
import com.eignex.combo.decisions.conditionedProblem
import com.eignex.combo.decisions.minimizeNative
import com.eignex.combo.bo.SearchTransform
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.combo.shared.LinearFeatureProjection
import com.eignex.combo.shared.WitnessRequest
import com.eignex.klause.formats.json.ProblemJson
import com.eignex.klause.ir.SourceRealId
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.result.QuadraticOptimizationResult
import com.eignex.klause.util.bigIntOf
import com.eignex.skema.SchemaJson
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.jsonObject
import kotlin.random.Random
import kotlin.test.*

class NativeProblemTest {
    @Test
    fun `canonical problem document survives model serialization without schema lowering`() {
        val problem = ProblemJson.default.decodeFromString("""
            {"bools":2,"ints":[{"lower":3000000000,"upper":3000000010,"values":[[3000000000,3000000002],[3000000009,3000000010]]}],
             "reals":[{}],"factors":[{"clause":{"literals":[0,2]}},
               {"table":{"vars":[0],"tuples":[[3000000000],[3000000010]]}}],
             "implied":[1],"symmetryBreaking":true}
        """.trimIndent())
        val definition = ModelDefinition(problem = problem)
        val document = Json.parseToJsonElement(SchemaJson.encodeToString(definition)).jsonObject
        assertEquals(ProblemJson.default.encodeToJsonElement(problem), document["problem"])
        val restored = SchemaJson.decodeFromString<ModelDefinition>(document.toString()).problem!!
        assertEquals(
            ProblemJson.default.encodeToJsonElement(problem), ProblemJson.default.encodeToJsonElement(restored),
        )
        assertEquals(problem.factors.map { it::class }, restored.factors.map { it::class })
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
        val problem = ProblemJson.default.decodeFromString("""
            {"ints":[{"lower":3000000000,"upper":3000000010,"values":[[3000000000,3000000000],[3000000010,3000000010]]}],
             "reals":[{}],"factors":[{"linear":{"vars":[0],"coefficients":[1],"op":"GE","bound":3000000005}}],"implied":[0]}
        """.trimIndent())
        val space = CompiledDecisionSpace.fromProblem(problem)
        val third = BigFraction.of(bigIntOf(1), bigIntOf(3))
        val given = context { set(SourceRealId(0), third) }
        val result = assertIs<QuadraticOptimizationResult.Optimal>(
            space.minimizeNative(space.coordinates.objective("ints.0"), given),
        )
        assertEquals(3000000010L, result.incumbent.sample.intValue(0))
        assertEquals(third, result.incumbent.sample.exactReals!![0])
        assertTrue(space.matches(result.incumbent.sample, given))
        val pinned = space.conditionedProblem(given)
        assertTrue(pinned.impliedFactorMask!![0])
        assertFalse(pinned.impliedFactorMask!![1])
    }

    @Test
    fun `native witness search feeds existing history and linear projections`() {
        val problem = ProblemJson.default.decodeFromString("""{"bools":1,"ints":[{"lower":0,"upper":3}]}""")
        val space = CompiledDecisionSpace.fromProblem(problem)
        val batch = HybridDescentSession.klause(space).witnesses(WitnessRequest(3), Random(42))
        assertEquals(3, batch.witnesses.size)
        for (sample in batch.witnesses) {
            assertEquals(2, SearchTransform(space).encode(BanditSample(sample), Context.Empty).size)
            assertEquals(2, LinearFeatureProjection(space).encode(BanditSample(sample), Context.Empty).size)
        }
    }
}
