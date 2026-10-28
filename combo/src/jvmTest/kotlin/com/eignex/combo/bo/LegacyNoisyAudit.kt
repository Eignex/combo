package com.eignex.combo.bo

import com.eignex.combo.bo.model.BatchForestConfig
import com.eignex.combo.bo.model.BatchSplitStrategy
import com.eignex.combo.decisions.Context
import java.io.File
import java.util.zip.GZIPInputStream
import kotlin.math.sqrt

/** Diagnose the archived noisy histories without rerunning or retuning their search trajectories. */
fun main() {
    val rows = GZIPInputStream(File("benchmarks/gp-comparison/traces.csv.gz").inputStream()).bufferedReader().use {
        it.readLines().drop(1).map { row -> row.split(',') }.filter { row ->
            row[0] == "noisy" && row[2] == "64" && row[3] in listOf("gp", "forest")
        }
    }
    val output = File("build/reports/noisy-bo/legacy").also { it.mkdirs() }
    File(output, "diagnostics.csv").writeText(
        "algorithm,seed,sd,history,rmse,latent_coverage,observation_coverage,small_gap,large_gap," +
            "small_regret,large_regret,pool_regret,top_rmse,top_latent_coverage\n",
    )
    for ((key, history) in rows.groupBy { it[1] to it[3] }) {
        val (seed, algorithm) = key
        val problem = refinementProblem("noisy")
        val optimizer = BayesianOptimizer(
            SearchTransform(problem.space),
            if (algorithm == "gp") {
                GaussianProcessSurrogate()
            } else {
                ForestSurrogate(
                    BatchForestConfig(splitStrategy = BatchSplitStrategy.Random, mtry = 2),
                )
            },
            outcome = OutcomeSpec(maximize = false, noise = ObservationNoise.Noisy, standardize = true),
            randomSeed = seed.toInt(),
            acquisitionOptimizer = noisySearch(problem),
        )
        for (row in history) {
            optimizer.train(
                continuousPoint(row[8].split(';').map(String::toDouble).toDoubleArray()),
                Context.Empty,
                row[5].toDouble(),
                noiseVariance = 0.0025 / 3.0,
            )
        }
        diagnoseNoisy(optimizer, problem, seed.toInt(), sqrt(0.0025 / 3.0), algorithm, output)
    }
}
