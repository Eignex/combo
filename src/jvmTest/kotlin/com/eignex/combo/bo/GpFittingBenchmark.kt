package com.eignex.combo.bo

import com.eignex.combo.bo.model.GaussianHyperparameterSearch
import com.eignex.combo.bo.model.GaussianMeanEstimation
import com.eignex.combo.bo.model.GaussianProcessConfig
import com.eignex.combo.bo.model.GaussianProcessFitting
import com.eignex.combo.bo.model.GaussianProcessRegularization
import com.eignex.combo.decisions.Context
import com.eignex.kumulant.stat.score.MseLossStat
import com.eignex.kumulant.stat.summary.MeanStat
import java.io.File
import java.util.zip.GZIPInputStream
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random

internal fun fittingVariants(): List<FittingVariant> {
    val gls = GaussianProcessFitting(meanEstimation = GaussianMeanEstimation.CovarianceWeighted)
    val prior = GaussianProcessRegularization()
    return listOf(
        FittingVariant("baseline24", 24),
        FittingVariant("baseline96"),
        FittingVariant("mean96", fitting = gls),
        FittingVariant("prior96", fitting = GaussianProcessFitting(regularization = prior)),
        FittingVariant(
            "multistart96",
            fitting = GaussianProcessFitting(search = GaussianHyperparameterSearch.Multistart),
        ),
        FittingVariant(
            "combined96",
            fitting = gls.copy(
                regularization = prior,
                search = GaussianHyperparameterSearch.Multistart,
            ),
        ),
    )
}

/** Optional fitting ablations. Objective truth only scores held-out predictions and final decisions. */
fun main(args: Array<String>) {
    val output = File("build/reports/gp-fitting/${args.firstOrNull() ?: "frozen"}").also { it.mkdirs() }
    when (args.firstOrNull() ?: "frozen") {
        "frozen" -> auditFitting(output)

        "sensitivity" -> auditFitting(output, sensitivity = true)

        "loop" -> {
            val names = args.getOrNull(1)?.split(',') ?: listOf("baseline24", "baseline96", "combined96")
            val seeds = args.getOrNull(2)?.split(',')?.map(String::toInt) ?: (1901..1906).toList()
            File(output, "outcomes.csv").writeText(
                "problem,seed,sd,batch,policy,calls,best_regret,recommendation_regret,model_fits," +
                    "suggestion_nanos,recommendation_nanos,acquisition_evaluations\n",
            )
            File(output, "traces.csv").writeText("problem,seed,sd,batch,policy,call,round,observed,truth,coordinates\n")
            for (name in listOf("quadratic2", "branin2", "multimodal3", "ellipsoid6")) {
                for (seed in seeds) {
                    for (sd in listOf(0.0, 0.03)) {
                        for (batch in listOf(1, 4)) {
                            for (variant in fittingVariants().filter { it.name in names }) {
                                runSuite(
                                    name, seed, sd, batch, "lognei-local", output,
                                    config = GaussianProcessConfig(hyperparameterTrials = variant.trials),
                                    fitting = variant.fitting, labelOverride = variant.name,
                                )
                            }
                        }
                    }
                    println("Fresh fitting $name/$seed")
                }
            }
        }

        else -> error("Expected frozen or loop")
    }
}

private fun auditFitting(output: File, sensitivity: Boolean = false) {
    val histories = GZIPInputStream(File("benchmarks/gp-refinement/traces.csv.gz").inputStream())
        .bufferedReader().use { reader -> reader.readLines().drop(1).map { it.split(',') } }
        .filter {
            it[4] == "lognei-local" && it[3] in listOf(
                "1",
                "4",
            )
        }.filter { !sensitivity || (it[2] == "0.03" && it[3] == "4") }.groupBy { it.take(5) }
    val scores = File(output, "scores.csv")
    scores.writeText(
        "problem,seed,sd,batch,policy,variant,regret,rmse,top_rmse,coverage,top_coverage," +
            "pool_gap,likelihood,score,mean,signal,noise,length_scales,replicate,fit_nanos\n",
    )
    for ((key, history) in histories) {
        val problem = suiteProblem(key[0])
        val projection = SearchTransform(problem.space)
        val d = projection.featureSize
        val random = Random(key[1].toInt() xor 73471)
        val heldout = List(4096) { continuousPoint(DoubleArray(d) { random.nextDouble() }) }
        val truth = heldout.map(problem.truth)
        val threshold = truth.sorted()[409]
        val variants = fittingVariants().filter {
            !sensitivity || it.fitting.search == GaussianHyperparameterSearch.Multistart
        }
        for (variant in variants) {
            for (replicate in 0 until if (sensitivity) 8 else 1) {
                lateinit var gp: FittedGaussianProcess
                val surrogate = Surrogate { data, stream ->
                    GaussianProcessSurrogate(
                        variant.fitting,
                        GaussianProcessConfig(hyperparameterTrials = variant.trials),
                    )
                        .fit(
                            data,
                            if (replicate == 0) stream else Random(stream.nextLong() xor replicate.toLong()),
                        ).also { gp = it }
                }
                val optimizer = BayesianOptimizer(
                    projection,
                    surrogate,
                    randomSeed = key[1].toInt(),
                    outcome = OutcomeSpec(maximize = false, noise = ObservationNoise.Noisy, standardize = true),
                )
                val observed = history.groupBy { it[6] }.values.flatMap { it.reversed() }.map { row ->
                    continuousPoint(row[9].split(';').map(String::toDouble).toDoubleArray()).also {
                        optimizer.train(
                            it,
                            Context.Empty,
                            row[7].toDouble(),
                            noiseVariance = key[2].toDouble().let { it * it },
                        )
                    }
                }
                val start = System.nanoTime()
                val model = optimizer.fittedModel()
                val fitNanos = System.nanoTime() - start
                val recommendation = optimizer.recommendOrThrow(
                    LocalAndRandomSearch(
                        projection,
                        uniformGpWitnesses(d),
                        candidateCount = 128,
                        maxAttempts = 256,
                        localSteps = 4,
                        neighbors = 8,
                    ),
                )
                val points = heldout + observed + recommendation
                val prediction = model.predict(featureRows(points.map { projection.encode(it) }, d))
                val mse = MseLossStat()
                val top = MseLossStat()
                val coverage = MeanStat()
                val topCoverage = MeanStat()
                for (i in heldout.indices) {
                    val covered = if (abs(prediction.means[i] - truth[i]) <= 2 * sqrt(prediction.variances[i])) {
                        1.0
                    } else {
                        0.0
                    }
                    mse.update(prediction.means[i], truth[i], 0L)
                    coverage.update(covered, 0L)
                    if (truth[i] <= threshold) {
                        top.update(prediction.means[i], truth[i], 0L)
                        topCoverage.update(covered, 0L)
                    }
                }
                val mean = prediction.means[points.lastIndex]
                val gap = mean - prediction.means.toDoubleArray().min()
                scores.appendText(
                    "${key.joinToString(",")},${variant.name},${problem.truth(recommendation)}," +
                        "${sqrt(mse.read(0L).mean)},${sqrt(top.read(0L).mean)},${coverage.read(0L).mean}," +
                        "${topCoverage.read(0L).mean},$gap,${gp.logMarginalLikelihood},${gp.logFitScore}," +
                        "${gp.parameters.mean},${gp.parameters.signalVariance},${gp.parameters.noiseVariance}," +
                        "${gp.parameters.lengthScales.joinToString(";")},$replicate,$fitNanos\n",
                )
            }
        }
        println("Frozen fitting ${key.joinToString("/")}")
    }
}
