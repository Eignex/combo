package com.eignex.combo.bench.diagnostics

import com.eignex.koblas.DenseVector
import com.eignex.kumulant.stat.summary.pairedMeanDifference
import com.eignex.lab.bench.BenchProtocol
import com.eignex.lab.bench.BenchResult
import com.eignex.lab.bench.BenchStatus
import kotlinx.serialization.json.*
import java.nio.file.Path
import kotlin.io.path.*
import kotlin.math.*
import kotlin.random.Random

/** Analyze the coordinator's accepted `lab cases JOB --json` snapshot, never directories of attempts. */
internal object AnalyzeDiagnostics {
    private data class PairRow(val case: String, val seed: String, val group: String, val arm: String,
        val metric: String, val baseline: Double, val candidate: Double)
    fun execute(args: Array<String>) {
        require(args.size in 2..3) { "analyze LAB_CASES.jsonl OUTPUT_DIRECTORY [BASELINE_ARM]" }
        val input = Path.of(args[0])
        val output = Path.of(args[1]).createDirectories()
        require(!output.resolve("paired.csv").exists()) { "Analysis output already exists" }
        val pairs = mutableListOf<PairRow>()
        var failed = 0
        var accepted = 0
        val seen = mutableSetOf<String>()
        input.useLines { lines -> lines.filter(String::isNotBlank).forEach { line ->
            val envelope = BenchProtocol.json.parseToJsonElement(line).jsonObject
            val record = envelope["record"]?.takeUnless { it is JsonNull }?.jsonObject
            if (record == null || envelope["status"]?.jsonPrimitive?.content != "DONE") { failed++; return@forEach }
            val value = record["labBench"] ?: record
            val result = BenchProtocol.json.decodeFromJsonElement(BenchResult.serializer(), value)
            if (result.status != BenchStatus.SUCCESS) { failed++; return@forEach }
            val id = result.caseId
            require(seen.add(id)) { "Duplicate accepted case id; analyze revision arms separately: $id" }
            accepted++
            val parts = id.split('/')
            require(parts.size >= 12)
            val preferred = args.getOrNull(2) ?: when (parts[0]) { "fitting" -> "baseline24"; "noise" -> "known"; "constrained" -> "random"; else -> "coordinate" }
            val metrics = result.metrics.associate { it.name to it.value }
            val available = result.metrics.map { it.name.substringBefore('.') }.distinct()
            require(args.size < 3 || preferred in available) { "Requested baseline $preferred is absent" }
            val base = preferred.takeIf { it in available } ?: available.first()
            val arms = result.metrics.map { it.name.substringBefore('.') }.distinct().filter { it != base }
            val group = (parts.take(3) + parts.subList(4, 11)).joinToString("/")
            val scores = metrics.keys.filter { it.startsWith("$base.") }.map { it.substringAfter('.') }
            for (arm in arms) for (metric in scores) {
                val a = metrics["$base.$metric"] ?: continue
                val b = metrics["$arm.$metric"] ?: continue
                pairs += PairRow(id, parts[3], group, arm, metric, a, b)
            }
        } }
        output.resolve("input-sha256.txt").writeText(fileDigest(input) + "\n")
        output.resolve("status.txt").writeText("accepted=$accepted\nfailed-or-unfinished=$failed\n" +
            "pairing=coordinator accepted cases; all suite arms within one worker attempt\n" +
            "bootstrap=2000 resamples of seed means; repeated rows and probes are not independent seeds\n")
        fun csv(value: String) = "\"" + value.replace("\"", "\"\"") + "\""
        output.resolve("paired.csv").bufferedWriter().use { writer ->
            writer.appendLine("case,seed,stratum,arm,metric,baseline,candidate,delta")
            for (p in pairs) writer.appendLine(listOf(p.case, p.seed, p.group, p.arm, p.metric, "${p.baseline}", "${p.candidate}", "${p.candidate-p.baseline}").joinToString(",", transform = ::csv))
        }
        output.resolve("summary.csv").bufferedWriter().use { writer ->
            writer.appendLine("stratum,arm,metric,rows,seeds,wins,losses,ties,mean_delta,standard_error,bootstrap_lo,bootstrap_hi")
            val groups = pairs.groupBy { Triple(it.group, it.arm, it.metric) } +
                pairs.groupBy { Triple(it.case.split('/').take(2).joinToString("/") + "/all-strata", it.arm, it.metric) }
            for ((key, rows) in groups.toSortedMap(compareBy<Triple<String,String,String>> { it.first }.thenBy { it.second }.thenBy { it.third })) {
                val seeds = rows.groupBy { it.seed }.toSortedMap().values
                val left = seeds.map { observations -> observations.map { it.candidate }.average() }
                val right = seeds.map { observations -> observations.map { it.baseline }.average() }
                val estimate = pairedMeanDifference(DenseVector.of(left.toDoubleArray()), DenseVector.of(right.toDoubleArray()))
                val deltas = left.indices.map { left[it] - right[it] }
                val random = Random(81877)
                val bootstrap = DoubleArray(2000) { List(deltas.size) { deltas[random.nextInt(deltas.size)] }.average() }.sorted()
                val wins = rows.count { it.candidate < it.baseline - 1e-12 }
                val losses = rows.count { it.candidate > it.baseline + 1e-12 }
                writer.appendLine(listOf(key.first,key.second,key.third,"${rows.size}","${seeds.size}","$wins","$losses","${rows.size-wins-losses}",
                    "${estimate.mean}","${estimate.standardError}","${bootstrap[49]}","${bootstrap[1949]}").joinToString(",", transform = ::csv))
            }
        }
        println("accepted=$accepted failed-or-unfinished=$failed pairs=${pairs.size} output=$output")
        require(accepted > 0 && failed == 0) { "Do not make complete-study claims from incomplete or failed cases" }
    }
}
