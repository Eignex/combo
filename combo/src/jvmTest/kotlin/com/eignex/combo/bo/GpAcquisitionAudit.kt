package com.eignex.combo.bo

import com.eignex.combo.decisions.Context
import com.eignex.koblas.DenseMatrix
import com.eignex.kumulant.math.nextNormal
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import java.io.File
import java.util.zip.GZIPInputStream
import kotlin.random.Random

/** Export frozen GP posteriors and captured normals for an independent BoTorch acquisition audit. */
fun main() {
    val output = File("build/reports/gp-refinement").also { it.mkdirs() }
    val archive = File("benchmarks/joint-batch/traces.csv.gz").inputStream().use { raw ->
        GZIPInputStream(raw).bufferedReader().readLines().drop(1).map { it.split(',') }
    }
    val fixtures = buildJsonArray {
        for (seed in 1501..1503) {
            for (historyPolicy in listOf("exclusion", "qnei")) {
                val problem = refinementProblem("noisy")
                val history = archive.filter {
                    it[0] == seed.toString() && it[1] == "0.03" && it[2] == historyPolicy
                }.take(24)
                val optimizer = BayesianOptimizer(
                    SearchTransform(problem.space),
                    GaussianProcessSurrogate(),
                    outcome = OutcomeSpec(maximize = false, noise = ObservationNoise.Noisy, standardize = true),
                    randomSeed = seed,
                )
                for (row in history) {
                    optimizer.train(
                        continuousPoint(doubleArrayOf(row[7].toDouble(), row[8].toDouble())),
                        Context.Empty,
                        row[5].toDouble(),
                        noiseVariance = 0.0009,
                    )
                }
                val model = optimizer.fittedModel()
                val completed = history.map {
                    listOf(
                        it[7].toDouble(),
                        it[8].toDouble(),
                    )
                }.distinct().sortedWith(coordinateOrder)
                val random = Random(seed xor 65431)
                val pool = List(6) { listOf(random.nextDouble(), random.nextDouble()) }.sortedWith(coordinateOrder)
                val pending = listOf(listOf(0.4, 0.6))
                for (noisy in listOf(false, true)) {
                    val base = if (noisy) (completed + pending).sortedWith(coordinateOrder) else emptyList()
                    val all = base + pool
                    val features = matrix(all)
                    val moments = model.predictJoint(features)!!
                    val incumbent = model.predict(matrix(completed)).means.toDoubleArray().min()
                    val factory = if (noisy) JointNoisyExpectedImprovement() else JointExpectedImprovement()
                    val objective = factory.create(
                        JointAcquisitionContext(
                            model,
                            incumbent,
                            OutcomeSpec(maximize = false),
                            matrix(completed),
                            if (noisy) matrix(pending) else matrix(emptyList()),
                        ),
                        Random(713),
                    )
                    val normal = Random(Random(713).nextLong())
                    val normals = List(base.size + 4) {
                        DoubleArray(256).also { row ->
                            for (i in row.indices step 2) {
                                row[i] = normal.nextNormal()
                                row[i + 1] = -row[i]
                            }
                        }.toList()
                    }
                    add(
                        buildJsonObject {
                            put("id", "$seed-$historyPolicy-${if (noisy) "qnei" else "qei"}")
                            put("baseline", base.size)
                            put("incumbent", -incumbent)
                            put(
                                "means",
                                Json.encodeToJsonElement(
                                    kotlinx.serialization.builtins.ListSerializer(Double.serializer()),
                                    moments.means.toDoubleArray().map { -it },
                                ),
                            )
                            val covariance = List(all.size) { i -> List(all.size) { j -> moments.covariance[i, j] } }
                            put("covariance", array(covariance))
                            put("normals", array(normals))
                            put(
                                "sets",
                                buildJsonArray {
                                    for (i in pool.indices) {
                                        fun record(indices: List<Int>) = buildJsonObject {
                                            val globalIndices = indices.map { it + base.size }
                                            put("indices", Json.parseToJsonElement(globalIndices.toString()))
                                            put("legacy", objective.evaluate(matrix(indices.map { pool[it] })))
                                        }
                                        add(record(listOf(i)))
                                        for (j in i + 1 until pool.size) add(record(listOf(i, j)))
                                        for (j in i + 1 until pool.size) {
                                            for (k in j + 1 until pool.size) {
                                                for (l in k + 1 until pool.size) add(record(listOf(i, j, k, l)))
                                            }
                                        }
                                    }
                                },
                            )
                        },
                    )
                }
            }
        }
    }
    File(
        output,
        "frozen.json",
    ).writeText(auditJson.encodeToString(kotlinx.serialization.json.JsonArray.serializer(), fixtures))
}
private val coordinateOrder = compareBy<List<Double>> { it[0] }.thenBy { it[1] }
private fun matrix(rows: List<List<Double>>): DenseMatrix = DenseMatrix.zero(rows.size, 2).also { values ->
    for (i in rows.indices) for (j in 0..1) values[i, j] = rows[i][j]
}
private fun array(rows: List<List<Double>>) = buildJsonArray {
    for (row in rows) add(buildJsonArray { for (x in row) add(kotlinx.serialization.json.JsonPrimitive(x)) })
}

private val auditJson = Json { prettyPrint = true }
