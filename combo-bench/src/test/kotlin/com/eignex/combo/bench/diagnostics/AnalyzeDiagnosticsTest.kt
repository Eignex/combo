package com.eignex.combo.bench.diagnostics

import com.eignex.lab.bench.*
import java.nio.file.Files
import kotlin.io.path.*
import kotlin.test.*

class AnalyzeDiagnosticsTest {
    @Test
    fun `analysis blocks repeated rows by seed and refuses duplicate accepted cases`() {
        val directory = Files.createTempDirectory("combo-analysis")
        try {
            fun row(seed: Int, repetition: Int, candidate: Double): String {
                val result = BenchResult("noise/loop/quadratic2/$seed/q1/sd0.03/r1/pfalse/baseline24/ctx0.75/fit0/$repetition",
                    metrics = listOf(BenchMetric("known.recommendation-regret", 2.0, "objective"),
                        BenchMetric("unknown.recommendation-regret", candidate, "objective")))
                return "{\"status\":\"DONE\",\"record\":" + BenchProtocol.json.encodeToString(BenchResult.serializer(), result) + "}"
            }
            val input = directory.resolve("cases.jsonl")
            input.writeText(listOf(row(7,1,1.0),row(7,2,3.0),row(8,1,1.0)).joinToString("\n"))
            val out = directory.resolve("analysis")
            // The real lab emits one JSON object per line; remove pretty-print newlines from this test fixture.
            input.writeText(input.readText().replace("\n", ""))
            val compact = kotlinx.serialization.json.Json { prettyPrint = false }
            input.writeText(listOf(row(7,1,1.0),row(7,2,3.0),row(8,1,1.0)).joinToString("\n") {
                compact.encodeToString(kotlinx.serialization.json.JsonElement.serializer(), BenchProtocol.json.parseToJsonElement(it))
            })
            AnalyzeDiagnostics.execute(arrayOf(input.toString(), out.toString()))
            val lines = out.resolve("summary.csv").readLines()
            assertTrue(lines.drop(1).all { it.contains("\"3\",\"2\"") && it.contains("\"-0.5\"") })
            input.appendText("\n" + compact.encodeToString(kotlinx.serialization.json.JsonElement.serializer(), BenchProtocol.json.parseToJsonElement(row(8,1,1.0))))
            assertFailsWith<IllegalArgumentException> { AnalyzeDiagnostics.execute(arrayOf(input.toString(), directory.resolve("duplicate").toString())) }
        } finally { directory.toFile().deleteRecursively() }
    }
}
