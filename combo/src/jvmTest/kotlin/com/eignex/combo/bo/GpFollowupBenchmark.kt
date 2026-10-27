package com.eignex.combo.bo

import com.eignex.combo.bo.model.GaussianProcessConfig
import com.eignex.combo.decisions.Context
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.kumulant.stat.score.MseLossStat
import com.eignex.kumulant.stat.summary.MeanStat
import com.sun.management.ThreadMXBean
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import java.io.File
import java.lang.management.ManagementFactory
import java.util.zip.GZIPInputStream
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random

/** Optional diagnostics: never use objective truth to fit a model or choose a recommendation. */
fun main(args: Array<String>) {
    when (args.firstOrNull()) {
        "recommendation" -> auditRecommendations()
        "convergence" -> auditConvergence()
        "score" -> scoreGpConvergence(followupOutput)
        "allocation" -> auditAllocations(args.getOrNull(1) ?: "0")
        else -> error("Expected recommendation, convergence, score, or allocation")
    }
}

private val followupOutput = File("build/reports/gp-followups").also { it.mkdirs() }
private fun archivedLines(path: String): List<List<String>> =
    GZIPInputStream(File(path).inputStream()).bufferedReader().use { reader ->
        reader.readLines().drop(1).map { it.split(',') }
    }

private fun auditRecommendations() {
    val histories = archivedLines("benchmarks/gp-refinement/traces.csv.gz").filter {
        it[0] in listOf("quadratic2", "ellipsoid6") &&
            it[4] in listOf("sequential-local", "nei-local", "lognei-local")
    }.groupBy { it.take(5) }
    val output = File(followupOutput, "recommendations.csv")
    output.writeText(
        "problem,seed,sd,batch,policy,trials,search,regret,predicted,best_evaluated,model_observed_regret," +
            "finite_pool_gap,rmse,top_rmse,coverage,top_coverage,log_likelihood,length_scales\n",
    )
    for ((key, history) in histories) {
        val (name, seedText, sdText) = key
        val seed = seedText.toInt()
        val sd = sdText.toDouble()
        val problem = suiteProblem(name)
        val projection = SearchTransform(problem.space)
        val d = projection.featureSize
        val random = Random(seed xor 73471)
        val heldout = List(4096) { continuousPoint(DoubleArray(d) { random.nextDouble() }) }
        for (trials in listOf(24, 96)) {
            lateinit var gp: FittedGaussianProcess
            val surrogate = Surrogate { data, stream ->
                GaussianProcessSurrogate(GaussianProcessConfig(hyperparameterTrials = trials)).fit(data, stream)
                    .also { gp = it }
            }
            val optimizer = BayesianOptimizer(
                projection,
                surrogate,
                outcome = OutcomeSpec(maximize = false, noise = ObservationNoise.Noisy, standardize = true),
                randomSeed = seed,
            )
            val completed = history.groupBy { it[6] }.values.flatMap { it.reversed() }
            val observed = completed.map { row ->
                continuousPoint(row[9].split(';').map(String::toDouble).toDoubleArray()).also {
                    optimizer.train(it, Context.Empty, row[7].toDouble(), noiseVariance = sd * sd)
                }
            }
            val model = optimizer.fittedModel()
            val recommendations = listOf(128, 1024).map { pool ->
                optimizer.recommendOrThrow(
                    LocalAndRandomSearch(
                        projection,
                        uniformGpWitnesses(d),
                        candidateCount = pool,
                        maxAttempts = pool * 2,
                        localSteps = if (pool == 128) 4 else 16,
                        neighbors = if (pool == 128) 8 else 16,
                    ),
                )
            }
            val points = heldout + observed + recommendations
            val prediction = model.predict(featureRows(points.map { projection.encode(it) }, d))
            val means = prediction.means
            val poolBest = means.toDoubleArray().min()
            val modelObserved = observed.indices.minBy { means[4096 + it] }
            val bestTruth = observed.minOf(problem.truth)
            val topThreshold = heldout.map(problem.truth).sorted()[409]
            val mse = MseLossStat()
            val top = MseLossStat()
            val coverage = MeanStat()
            val topCoverage = MeanStat()
            for (i in heldout.indices) {
                val truth = problem.truth(heldout[i])
                val covered = if (abs(means[i] - truth) <= 2 * sqrt(prediction.variances[i])) 1.0 else 0.0
                mse.update(means[i], truth, 0L)
                coverage.update(covered, 0L)
                if (truth <= topThreshold) {
                    top.update(means[i], truth, 0L)
                    topCoverage.update(covered, 0L)
                }
            }
            for (i in recommendations.indices) {
                val mean = means[points.size - 2 + i]
                output.appendText(
                    "${key.joinToString(",")},$trials,${if (i == 0) "standard" else "large"}," +
                        "${problem.truth(
                            recommendations[i],
                        )},$mean,$bestTruth,${problem.truth(observed[modelObserved])}," +
                        "${mean - poolBest},${sqrt(mse.read(0L).mean)},${sqrt(top.read(0L).mean)}," +
                        "${coverage.read(0L).mean},${topCoverage.read(0L).mean},${gp.logMarginalLikelihood}," +
                        "${gp.parameters.lengthScales.joinToString(";")}\n",
                )
            }
        }
        println("Frozen recommendation ${key.joinToString("/")}")
    }
}

private fun auditConvergence() {
    val fixtures = GZIPInputStream(File("benchmarks/gp-refinement/frozen.json.gz").inputStream())
        .bufferedReader().use { Json.parseToJsonElement(it.readText()).jsonArray.toList() } + extendedFixtures()
    File(followupOutput, "posteriors.json").writeText(
        kotlinx.serialization.json.JsonArray(fixtures).toString(),
    )
    val output = File(followupOutput, "convergence.csv")
    output.writeText("fixture,q,replicate,samples,set,raw,log\n")
    for (element in fixtures) {
        val fixture = element.jsonObject
        val id = fixture.getValue("id").jsonPrimitive.content
        val baseline = fixture.getValue("baseline").jsonPrimitive.int
        val means = fixture.getValue("means").jsonArray.map { it.jsonPrimitive.double }
        val covariance = fixture.getValue("covariance").jsonArray.map { row ->
            row.jsonArray.map { it.jsonPrimitive.double }
        }
        val model = object : FittedSurrogate {
            override val uncertaintyKind = UncertaintyKind.LatentPosterior
            override fun predict(features: DenseMatrix): PredictiveMoments {
                val joint = predictJoint(features)
                return PredictiveMoments(
                    joint.means,
                    DenseVector.of(
                        DoubleArray(features.rows) {
                            joint.covariance[it, it]
                        },
                    ),
                )
            }
            override fun predictJoint(features: DenseMatrix): JointPredictiveMoments {
                val indices = List(features.rows) { features[it, 0].toInt() }
                return JointPredictiveMoments(
                    DenseVector.of(DoubleArray(indices.size) { means[indices[it]] }),
                    DenseMatrix.zero(indices.size, indices.size).also { values ->
                        for (i in indices.indices) {
                            for (j in indices.indices) {
                                values[i, j] = covariance[indices[i]][indices[j]]
                            }
                        }
                    },
                )
            }
            override fun drawFunction(random: Random): SampledFunction = error("unused")
        }
        fun features(indices: List<Int>): DenseMatrix = DenseMatrix.zero(indices.size, 1).also { values ->
            for (i in indices.indices) values[i, 0] = indices[i].toDouble()
        }
        val context = JointAcquisitionContext(
            model,
            fixture.getValue("incumbent").jsonPrimitive.double,
            OutcomeSpec(),
            features((0 until baseline).toList()),
        )
        for (replicate in 0..9) {
            val counts = if (replicate >= 8) listOf(65536) else listOf(256, 1024, 4096)
            for (samples in counts) {
                val noisy = id.endsWith("qnei")
                val raw = (if (noisy) JointNoisyExpectedImprovement(samples) else JointExpectedImprovement(samples))
                    .create(context, Random(7123 + replicate))
                val log = (
                    if (noisy) {
                        JointLogNoisyExpectedImprovement(
                            samples,
                        )
                    } else {
                        JointLogExpectedImprovement(samples)
                    }
                    )
                    .create(context, Random(7123 + replicate))
                for (q in listOf(1, 2, 4)) {
                    for (set in subsets((baseline until means.size).toList(), q)) {
                        val points = features(set)
                        output.appendText(
                            "$id,$q,$replicate,$samples,${set.joinToString(";")}," +
                                "${raw.evaluate(points)},${log.evaluate(points)}\n",
                        )
                    }
                }
            }
        }
        println("Frozen sampling $id")
    }
}

private fun extendedFixtures(): List<JsonElement> {
    val histories = archivedLines("benchmarks/gp-refinement/traces.csv.gz").filter {
        it[1] in listOf("1701", "1702") && it[2] == "0.03" && it[3] == "4" && it[4] == "lognei-local"
    }.groupBy { it.take(5) }
    return histories.map { (key, history) ->
        val problem = suiteProblem(key[0])
        val projection = SearchTransform(problem.space)
        val optimizer = BayesianOptimizer(
            projection,
            GaussianProcessSurrogate(),
            randomSeed = key[1].toInt(),
            outcome = OutcomeSpec(maximize = false, noise = ObservationNoise.Noisy, standardize = true),
        )
        val points = history.take(24).groupBy { it[6] }.values.flatMap { it.reversed() }.map { row ->
            continuousPoint(row[9].split(';').map(String::toDouble).toDoubleArray()).also {
                optimizer.train(it, Context.Empty, row[7].toDouble(), noiseVariance = 0.0009)
            }
        }
        val baseline = points.map { projection.encode(it).toDoubleArray().toList() }.distinct()
            .sortedWith(featureOrdering)
        val random = Random(key[1].toInt() xor 65431)
        val pool = List(6) { List(projection.featureSize) { random.nextDouble() } }.sortedWith(featureOrdering)
        val moments = optimizer.fittedModel().predictJoint(featureMatrix(baseline + pool, projection.featureSize))!!
        fun array(rows: List<List<Double>>) = buildJsonArray {
            for (row in rows) add(buildJsonArray { for (value in row) add(JsonPrimitive(value)) })
        }
        buildJsonObject {
            put("id", "${key[0]}-${key[1]}-qnei")
            put("baseline", baseline.size)
            put("incumbent", 0.0)
            put("means", buildJsonArray { for (value in moments.means.toDoubleArray()) add(JsonPrimitive(-value)) })
            put(
                "covariance",
                array(
                    List(baseline.size + pool.size) { i ->
                        List(baseline.size + pool.size) { j -> moments.covariance[i, j] }
                    },
                ),
            )
        }
    }
}

private fun subsets(values: List<Int>, size: Int): List<List<Int>> = when {
    size == 0 -> listOf(emptyList())
    values.size < size -> emptyList()
    else -> values.indices.flatMap { i -> subsets(values.drop(i + 1), size - 1).map { listOf(values[i]) + it } }
}

private fun auditAllocations(fork: String) {
    val output = File(followupOutput, "allocation-$fork.csv")
    output.writeText("baseline,q,kind,implementation,repeat,evaluations,nanos,bytes,checksum\n")
    val bean = ManagementFactory.getThreadMXBean() as ThreadMXBean
    bean.isThreadAllocatedMemoryEnabled = true
    val thread = Thread.currentThread().threadId()
    for (baseline in listOf(16, 64, 128)) {
        for (q in listOf(1, 4)) {
            val random = Random(851)
            val training = featureMatrix(List(baseline) { listOf(random.nextDouble(), random.nextDouble()) }, 2)
            val gp = GaussianProcessSurrogate(GaussianProcessConfig(hyperparameterTrials = 0)).fit(
                TrainingData(
                    training,
                    DenseVector.of(DoubleArray(baseline) { training[it, 0] + training[it, 1] }),
                    DenseVector.of(DoubleArray(baseline) { 1.0 }),
                    DenseVector.of(DoubleArray(baseline) { 0.001 }),
                ),
                Random(1),
            )
            val context = JointAcquisitionContext(gp, null, OutcomeSpec(), training)
            val candidates = List(20) { featureMatrix(List(q) { listOf(random.nextDouble(), random.nextDouble()) }, 2) }
            repeat(530) { iteration ->
                for (log in if (iteration % 2 == 0) listOf(false, true) else listOf(true, false)) {
                    val checksums = mutableListOf<Double>()
                    for (reference in if (iteration % 2 == 0) listOf(false, true) else listOf(true, false)) {
                        val objective = if (reference) {
                            legacyGpImprovement(context, Random(77), 256, 0.0, true, if (log) 1e-6 else null)
                        } else {
                            (if (log) JointLogNoisyExpectedImprovement() else JointNoisyExpectedImprovement())
                                .create(context, Random(77))
                        }
                        val before = bean.getThreadAllocatedBytes(thread)
                        val start = System.nanoTime()
                        var checksum = 0.0
                        for (points in candidates) checksum += objective.evaluate(points)
                        val nanos = System.nanoTime() - start
                        val bytes = bean.getThreadAllocatedBytes(thread) - before
                        checksums += checksum
                        if (iteration >= 500) {
                            val kind = if (log) "log" else "raw"
                            val implementation = if (reference) "reference" else "current"
                            output.appendText(
                                "$baseline,$q,$kind,$implementation," +
                                    "${iteration - 500},20,$nanos,$bytes,$checksum\n",
                            )
                        }
                    }
                    check(abs(checksums[0] - checksums[1]) <= 1e-12) {
                        "baseline=$baseline q=$q log=$log iteration=$iteration checksums=$checksums"
                    }
                }
            }
        }
    }
}
