package com.eignex.combo.bench

import com.eignex.lab.bench.BenchContext
import com.eignex.lab.bench.BenchRequest
import java.nio.file.Files
import kotlin.io.path.readLines
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class LabAdapterTest {
    @Test
    fun `planning keeps every arm in each workload seed repetition row`() {
        val cases = ComboLabSuite.plan(BenchRequest(mapOf(
            "workloads" to "quadratic,noisy", "seeds" to "3,11", "repetitions" to "2")))
        assertEquals(8, cases.size)
        assertEquals(8, cases.map { it.id }.distinct().size)
        assertTrue(cases.all { it.parameters["algorithms"] == "gp,forest,random" })
        assertTrue(cases.all { it.resources.requiredTags.isEmpty() })
        assertEquals(cases, ComboLabSuite.plan(BenchRequest(mapOf(
            "workloads" to "quadratic,noisy", "seeds" to "3,11", "repetitions" to "2"))))
    }

    @Test
    fun `invalid requests fail before execution`() {
        for (parameters in listOf(
            mapOf("workloads" to "unknown"), mapOf("algorithms" to "gp,gp"),
            mapOf("seeds" to "3,3"), mapOf("seeds" to "abc"), mapOf("budget" to "0"),
            mapOf("repetitions" to "0"), mapOf("memory-mib" to "1024"),
            mapOf("timeout-seconds" to "-1"), mapOf("smoke" to "yes"), mapOf("typo" to "true"),
        )) assertFailsWith<IllegalArgumentException> { ComboLabSuite.plan(BenchRequest(parameters)) }
    }

    @Test
    fun `all workloads complete finite paired traces and seeded replay`() {
        val directory = Files.createTempDirectory("combo-bench-test")
        try {
            val cases = ComboLabSuite.plan(BenchRequest(mapOf("budget" to "4", "seeds" to "3")))
            for (case in cases) {
                val first = BenchContext(directory.resolve(case.id).resolve("first"))
                val result = ComboLabSuite.run(case, first)
                val second = BenchContext(directory.resolve(case.id).resolve("second"))
                val replay = ComboLabSuite.run(case, second)
                assertEquals(15, result.metrics.size)
                assertEquals(4, result.artifacts.size)
                assertTrue(result.metrics.all { it.value.isFinite() })
                assertEquals(result.metrics.filterNot { it.name.endsWith(".elapsed") },
                    replay.metrics.filterNot { it.name.endsWith(".elapsed") })
                for (algorithm in listOf("gp", "forest", "random")) {
                    val rows = first.outputDirectory.resolve("report/$algorithm.csv").readLines()
                    assertEquals(5, rows.size)
                    val regrets = rows.drop(1).map { it.split(',')[4].toDouble() }
                    assertTrue(regrets.zipWithNext().all { (a, b) -> b <= a })
                    assertEquals(rows, second.outputDirectory.resolve("report/$algorithm.csv").readLines())
                }
                assertEquals(0.0, result.metrics.single { it.name == "random.model-fits" }.value)
                assertTrue(result.metrics.single { it.name == "gp.model-fits" }.value > 0)
                assertTrue(result.metrics.single { it.name == "forest.model-fits" }.value > 0)
            }
        } finally {
            directory.toFile().deleteRecursively()
        }
    }

    @Test
    fun `tampered frozen cases are rejected`() {
        val case = ComboLabSuite.plan(BenchRequest(mapOf("smoke" to "true"))).single()
        val directory = Files.createTempDirectory("combo-bench-invalid")
        try {
            assertFailsWith<IllegalArgumentException> {
                ComboLabSuite.run(case.copy(id = "wrong"), BenchContext(directory))
            }
            assertFailsWith<IllegalArgumentException> {
                ComboLabSuite.run(case.copy(resources = case.resources.copy(cpus = 2)), BenchContext(directory))
            }
        } finally {
            directory.toFile().deleteRecursively()
        }
    }
}
