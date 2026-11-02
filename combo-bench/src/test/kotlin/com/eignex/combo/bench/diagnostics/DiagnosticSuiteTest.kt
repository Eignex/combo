package com.eignex.combo.bench.diagnostics

import com.eignex.lab.bench.*
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.*
import kotlin.test.*

class DiagnosticSuiteTest {
    private fun request(study: String, vararg extra: Pair<String, String>) = BenchRequest(mapOf(
        "study" to study, "smoke" to "true", "noise-sd" to "0.03", "fit-trials" to "4", "workloads" to "quadratic2") + extra)
    @Test
    fun `plans freeze complete paired rows and reject invalid settings`() {
        val request = request("noise", "seeds" to "7,8", "batch" to "1,4", "pending" to "false,true")
        val cases = DiagnosticSuite.plan(request)
        assertEquals(8, cases.size)
        assertEquals(cases, DiagnosticSuite.plan(request))
        assertTrue(cases.all { it.parameters["arms"] == "known,unknown,partial" && it.inputs.single().sha256 != null && it.resources.requiredTags.isEmpty() })
        for (extra in listOf("batch" to "0", "noise-sd" to "NaN", "seeds" to "1,01", "grid-nodes" to "4", "arms" to "mixture", "typo" to "1")) {
            assertFailsWith<IllegalArgumentException> { DiagnosticSuite.plan(request("noise", extra)) }
        }
    }
    @Test
    fun `frozen and asynchronous noisy cases retain complete finite replayable captures`() {
        val directory = Files.createTempDirectory("combo-diagnostics")
        try {
            for (phase in listOf("frozen", "loop")) {
                val case = DiagnosticSuite.plan(request("noise", "phase" to phase, "batch" to "4", "pending" to "true", "budget" to "24", "replicates" to "2")).single()
                val first = DiagnosticSuite.run(case, BenchContext(directory.resolve("$phase-first")))
                val second = DiagnosticSuite.run(case, BenchContext(directory.resolve("$phase-second")))
                assertTrue(first.metrics.all { it.value.isFinite() })
                assertTrue(first.metrics.filter { it.name.endsWith(".evaluations") }.all { it.value == 24.0 })
                if (phase == "loop") assertTrue(first.metrics.filter { it.name.endsWith(".max-pending") }.all { it.value == 8.0 })
                val replay = second.metrics.associateBy { it.name }
                for (metric in first.metrics.filterNot { it.name.endsWith("-seconds") }) {
                    assertEquals(metric.value, replay.getValue(metric.name).value, 1e-12, metric.name)
                }
                assertEquals(11, first.artifacts.size)
            }
        } finally { directory.toFile().deleteRecursively() }
    }

    @Test
    fun `constrained cases retain exact checked witnesses and cancelled work stays failed`() {
        val directory = Files.createTempDirectory("combo-constrained")
        try {
            val cases = DiagnosticSuite.plan(BenchRequest(mapOf("study" to "constrained", "smoke" to "true", "noise-sd" to "0.0", "fit-trials" to "0", "budget" to "12")))
            assertEquals(4, cases.size)
            for ((index, case) in cases.withIndex()) {
                val result = DiagnosticSuite.run(case, BenchContext(directory.resolve("case-$index")))
                assertEquals(BenchStatus.SUCCESS, result.status, result.message)
                assertTrue(result.metrics.filter { it.name.endsWith(".completed-source-checks") }.all { it.value == 12.0 })
                val problem = DiagnosticProblem(case.parameters.getValue("workload"))
                assertEquals(true, problem.check(problem.point(kotlin.random.Random(7)).sample))
            }
            Thread.currentThread().interrupt()
            try {
                val result = DiagnosticSuite.run(cases.first(), BenchContext(directory.resolve("cancelled")))
                assertEquals(BenchStatus.FAILED, result.status)
                assertEquals("benchmark cancelled", result.message)
                assertTrue(result.artifacts.any { it.path.endsWith("failure.txt") })
            } finally { Thread.interrupted() }
        } finally { directory.toFile().deleteRecursively() }
    }
    @Test
    fun `prepared fixtures survive relocation and reject changed inputs and case identities`() {
        val case = DiagnosticSuite.plan(request("fitting", "phase" to "frozen", "arms" to "coordinate")).single()
        val directory = Files.createTempDirectory("combo-relocated")
        try {
            val root = directory.resolve("source")
            val input = case.inputs.single()
            val relocated = root.resolve(input.path)
            relocated.parent.createDirectories()
            Path.of(input.path).copyTo(relocated)
            val result = DiagnosticSuite.run(case, BenchContext(directory.resolve("out"), worktree = root))
            assertEquals(BenchStatus.SUCCESS, result.status)
            assertFailsWith<IllegalArgumentException> { DiagnosticSuite.run(case.copy(id = "tampered"), BenchContext(directory.resolve("bad-id"), worktree = root)) }
            relocated.appendText("bad bytes")
            assertFailsWith<IllegalArgumentException> { DiagnosticSuite.run(case, BenchContext(directory.resolve("bad-input"), worktree = root)) }
        } finally { directory.toFile().deleteRecursively() }
    }
}
