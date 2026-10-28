package com.eignex.combo.bo

import com.eignex.kumulant.math.SobolSequence
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.util.zip.GZIPInputStream
import kotlin.random.Random

/** Optional sampling comparison over frozen posteriors and fresh, paired external-evaluation budgets. */
fun main(args: Array<String>) {
    when (args.firstOrNull()) {
        "frozen" -> compareFrozenSampling()
        "reference" -> exportQmcReference()
        "loop" -> compareSamplingLoops(args.getOrNull(1).orEmpty())
        else -> error("Expected frozen, reference, or loop")
    }
}

private val samplingOutput = File("build/reports/gp-qmc").also { it.mkdirs() }
private fun fixtures(path: String): JsonArray = GZIPInputStream(File(path).inputStream()).bufferedReader().use {
    Json.parseToJsonElement(it.readText()).jsonArray
}
private fun fixtureModel(fixture: JsonObject): MatrixPosterior = MatrixPosterior(
    fixture.getValue("means").jsonArray.map { it.jsonPrimitive.double },
    fixture.getValue("covariance").jsonArray.map { row -> row.jsonArray.map { it.jsonPrimitive.double } },
)
private fun fixtureContext(fixture: JsonObject): JointAcquisitionContext = JointAcquisitionContext(
    fixtureModel(fixture),
    fixture.getValue("incumbent").jsonPrimitive.double,
    OutcomeSpec(),
    jointPoints(*(0 until fixture.getValue("baseline").jsonPrimitive.int).map(Int::toDouble).toDoubleArray()),
)
private fun sets(values: List<Int>, count: Int): List<List<Int>> = when {
    count == 0 -> listOf(emptyList())
    values.size < count -> emptyList()
    else -> values.indices.flatMap { i -> sets(values.drop(i + 1), count - 1).map { listOf(values[i]) + it } }
}

private fun compareFrozenSampling() {
    val fixtures = fixtures("benchmarks/gp-followups/posteriors.json.gz")
    for (sampling in JointNormalSampling.entries) {
        val directory = File(samplingOutput, sampling.name).also { it.mkdirs() }
        val output = File(directory, "convergence.csv")
        output.writeText("fixture,q,replicate,samples,set,raw,log\n")
        File("benchmarks/gp-followups/sobol.csv").copyTo(File(directory, "sobol.csv"), overwrite = true)
        for (element in fixtures) {
            val fixture = element.jsonObject
            val id = fixture.getValue("id").jsonPrimitive.content
            val context = fixtureContext(fixture)
            val candidates = (context.baseline.rows until fixture.getValue("means").jsonArray.size).toList()
            for (replicate in 0..7) {
                for (samples in listOf(256, 1024, 4096)) {
                    val noisy = id.endsWith("qnei")
                    val raw = (
                        if (noisy) {
                            JointNoisyExpectedImprovement(sampling, samples)
                        } else {
                            JointExpectedImprovement(sampling, samples)
                        }
                        ).create(context, Random(7123 + replicate))
                    val log = (
                        if (noisy) {
                            JointLogNoisyExpectedImprovement(sampling, samples)
                        } else {
                            JointLogExpectedImprovement(sampling, samples)
                        }
                        ).create(context, Random(7123 + replicate))
                    for (q in listOf(1, 2, 4)) {
                        for (indices in sets(candidates, q)) {
                            val features = jointPoints(*indices.map(Int::toDouble).toDoubleArray())
                            output.appendText(
                                "$id,$q,$replicate,$samples,${indices.joinToString(";")}," +
                                    "${raw.evaluate(features)},${log.evaluate(features)}\n",
                            )
                        }
                    }
                }
            }
            println("Sampling $sampling frozen $id")
        }
        scoreGpConvergence(directory)
    }
}

private fun exportQmcReference() {
    val exported = fixtures("benchmarks/gp-followups/frozen-q4.json.gz").map { element ->
        val fixture = element.jsonObject
        val context = fixtureContext(fixture)
        val baseline = context.baseline.rows
        val sequence = SobolSequence(baseline + 4, Random(713).nextLong())
        val normals = Array(baseline + 4) { DoubleArray(256) }
        val point = DoubleArray(baseline + 4)
        repeat(256) { draw ->
            sequence.nextNormal(point)
            for (coordinate in point.indices) normals[coordinate][draw] = point[coordinate]
        }
        val noisy = fixture.getValue("id").jsonPrimitive.content.endsWith("qnei")
        val raw = (
            if (noisy) {
                JointNoisyExpectedImprovement(JointNormalSampling.ScrambledSobol)
            } else {
                JointExpectedImprovement(JointNormalSampling.ScrambledSobol)
            }
            ).create(context, Random(713))
        val records = fixture.getValue("sets").jsonArray.map { case ->
            val row = case.jsonObject
            val indices = row.getValue("indices").jsonArray.map { it.jsonPrimitive.int }
            JsonObject(
                row + (
                    "legacy" to JsonPrimitive(
                        raw.evaluate(jointPoints(*indices.map(Int::toDouble).toDoubleArray())),
                    )
                    ),
            )
        }
        JsonObject(
            fixture + mapOf(
                "normals" to buildJsonArray {
                    for (row in normals) add(buildJsonArray { for (value in row) add(JsonPrimitive(value)) })
                },
                "sets" to JsonArray(records),
            ),
        )
    }
    File(samplingOutput, "frozen.json").writeText(JsonArray(exported).toString())
}

private fun compareSamplingLoops(seedText: String) {
    val seeds = seedText.takeIf { it.isNotBlank() }?.split(',')?.map(String::toInt) ?: (1801..1806).toList()
    val output = File(samplingOutput, "validation").also { it.mkdirs() }
    File(output, "outcomes.csv").writeText(
        "problem,seed,sd,batch,policy,calls,best_regret,recommendation_regret,model_fits," +
            "suggestion_nanos,recommendation_nanos,acquisition_evaluations\n",
    )
    File(output, "traces.csv").writeText("problem,seed,sd,batch,policy,call,round,observed,truth,coordinates\n")
    for (name in listOf("quadratic2", "branin2", "multimodal3", "ellipsoid6")) {
        for (seed in seeds) {
            for (sd in listOf(0.0, 0.03)) {
                for (batch in listOf(1, 4)) {
                    for (sampling in JointNormalSampling.entries) {
                        runSuite(name, seed, sd, batch, "lognei-local", output, sampling)
                    }
                }
            }
            println("Fresh sampling $name seed=$seed")
        }
    }
}
