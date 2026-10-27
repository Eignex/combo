package com.eignex.combo.model

import com.eignex.combo.decisions.DecisionSpaceDef
import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.backtrack.BacktrackSolver
import com.eignex.klause.compile.compile
import com.eignex.klause.model.*
import com.eignex.klause.propagation.bake
import com.eignex.klause.schema.*
import com.eignex.klause.solver.SolveResult
import com.eignex.kumulant.schema.BandResult
import com.eignex.kumulant.schema.ops.band
import com.eignex.kumulant.schema.spec.Variance
import com.eignex.skema.SchemaJson
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.*
import kotlin.test.*

class StatisticalLoweringTest {
    private val latency = StatisticalInt("latency")
    private val snapshot = mapOf("timings" to encodeStatisticResult(BandResult(100.0, 20.0, 1.0, 80.0, 120.0)))

    private fun model(constraint: BoolExpr): ModelDefinition = ModelDefinition(
        space = DecisionSpaceDef(constraints = mapOf("limit" to constraint)),
        statistics = mapOf("timings" to Variance.band(1.0)),
        inputs = mapOf("latency" to StatisticalInput("timings", projections = mapOf("mean" to listOf("center")))),
    )

    private fun feasible(model: ModelDefinition, data: Map<String, JsonElement> = snapshot): Boolean {
        val resolved = model.resolveStatistics(data)
        val compiled = resolved.schema.compile()
        return BacktrackSolver(compiled.problem.bake()).session().use { it.solve(BacktrackParams()) is SolveResult.Sat }
    }

    @Test
    fun `comparisons account for the entire interval and reverse under negation`() {
        for ((constraint, expected) in listOf(
            (latency lt 100) to false,
            (latency le 120) to true,
            (latency gt 100) to false,
            (latency ge 80) to true,
            (latency eq 100) to false,
            (latency ne 100) to false,
            (latency ne 150) to true,
            (!(latency lt 80)) to true,
            ((-latency) le -80) to true,
            (100 lt latency) to false,
            ((latency - latency) eq 0) to true,
        )) assertEquals(expected, feasible(model(constraint)), constraint.toString())
    }

    @Test
    fun `nonlinear uncertainty requires a projection but native nonlinear expressions remain available`() {
        val workers = IntHandle("workers", 1, 4)
        val ambiguous = model((latency * workers) le 240).copy(
            space = DecisionSpaceDef(
                variables = mapOf("workers" to IntSpec(1, 4)),
                constraints = mapOf("limit" to ((latency * workers) le 240)),
            ),
        )
        val error = assertFailsWith<IllegalArgumentException> { ambiguous.resolveStatistics(snapshot) }
        assertTrue(error.message.orEmpty().contains(".upper"))
        val explicit = ambiguous.copy(space = ambiguous.space!!.copy(
            constraints = mapOf("limit" to ((latency.upper * workers) le 240)),
        ))
        assertTrue(feasible(explicit))
        val impossible = explicit.copy(space = explicit.space!!.copy(constraints = mapOf(
            "limit" to ((latency.upper * workers) le 240),
            "minimum_workers" to (workers ge 3),
        )))
        assertFalse(feasible(impossible))
    }

    @Test
    fun `uncertain disjunction is rejected rather than distributing the universal quantifier`() {
        val expression = (latency le 80) or (latency ge 120)
        assertFailsWith<IllegalArgumentException> { model(expression).resolveStatistics(snapshot) }
        assertTrue(feasible(model((latency.lower le 80) or (latency.upper ge 120))))
    }

    @Test
    fun `fractional real outputs retain their values without integer rounding`() {
        val real = StatisticalReal("latency")
        val definition = model(real lt 120.5)
        assertTrue(feasible(definition))
        assertFalse(feasible(model(real lt 119.5)))
        val fractional = mapOf("timings" to encodeStatisticResult(BandResult(90.5, 9.25, 1.0, 81.25, 99.75)))
        assertTrue(feasible(model(real lt 100.0), fractional))
        assertFailsWith<IllegalArgumentException> { model(latency lt 100).resolveStatistics(fractional) }
    }

    @Test
    fun `bound inputs remain native expressions through JSON and qualification`() {
        val original = model(latency le 120)
        val encoded = SchemaJson.encodeToString(original)
        assertFalse("StatisticalInt" in encoded)
        assertTrue("\"ref\":\"latency\"" in encoded)
        val restored = SchemaJson.decodeFromString<ModelDefinition>(encoded)
        assertEquals(original, restored)
        assertTrue(feasible(restored))
        val nested = original.copy(
            space = DecisionSpaceDef(spaces = mapOf("child" to original.space!!)),
            inputs = mapOf("child.latency" to original.inputs.getValue("latency")),
        )
        assertTrue(feasible(nested))
    }

    @Test
    fun `unavailable and inverted interval outputs cannot become convenient solver choices`() {
        for (data in listOf(
            mapOf("timings" to buildJsonObject { put("lower", 120); put("upper", 80) }),
            mapOf("timings" to buildJsonObject { put("lower", 80); put("upper", JsonNull) }),
            mapOf("timings" to buildJsonObject { put("lower", 80) }),
        )) assertFailsWith<IllegalArgumentException> { model(latency lt 100).resolveStatistics(data) }
    }

    @Test
    fun `one joint native snapshot supplies multiple views and remains frozen after observation`() {
        val definition = SchemaJson.decodeFromString<ModelDefinition>("""
            {"space":{"constraints":{"ordered":{"lt":[{"ref":"left"},{"ref":"right"}]}}},
             "statistics":{"joint":{"covariance":{}}},
             "inputs":{
               "left":{"statistic":"joint","value":"meanX"},
               "right":{"statistic":"joint","value":"meanY"}}}
        """.trimIndent())
        ModelStatistics(definition).use { statistics ->
            assertFailsWith<InsufficientStatisticalEvidence> { statistics.resolve() }
            statistics.observe("joint", buildJsonObject { put("x", 2); put("y", 3) })
            val frozen = statistics.resolve()
            assertEquals(2.0, frozen.inputs.getValue("left.meanX").jsonPrimitive.double)
            assertEquals(3.0, frozen.inputs.getValue("right.meanY").jsonPrimitive.double)
            statistics.observe("joint", buildJsonObject { put("x", 20); put("y", 31) })
            assertEquals(2.0, frozen.inputs.getValue("left.meanX").jsonPrimitive.double)
            assertEquals(11.0, statistics.resolve().inputs.getValue("left.meanX").jsonPrimitive.double)
        }
    }
}
