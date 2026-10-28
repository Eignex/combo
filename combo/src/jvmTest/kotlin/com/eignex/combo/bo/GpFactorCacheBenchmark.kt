package com.eignex.combo.bo

import com.eignex.combo.bo.model.GaussianProcessConfig
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.koblas
import com.eignex.kumulant.math.factorCholesky
import com.eignex.kumulant.math.nextNormal
import com.eignex.kumulant.stat.score.MaeLossStat
import com.eignex.kumulant.stat.summary.MeanStat
import java.io.File
import kotlin.math.abs
import kotlin.random.Random

/** Whole-round posterior/factor microbenchmark; cached binding time is included and GP fitting is excluded. */
fun main(args: Array<String>) {
    val suffix = args.firstOrNull()?.takeUnless { it == "0" }?.let { "-$it" } ?: ""
    val output = File("build/reports/gp-refinement/factor-cache$suffix.csv")
    output.parentFile.mkdirs()
    output.writeText(
        "baseline,q,repeat,sets,full_nanos,cached_nanos,full_factor_order,cached_baseline_order," +
            "cached_candidate_order,baseline_factorizations,candidate_factorizations,mae\n",
    )
    for (size in listOf(16, 64, 128)) {
        for (q in listOf(1, 4)) {
            val random = Random(851)
            val baseline = featureMatrix(
                List(size) { listOf(random.nextDouble(), random.nextDouble()) }.sortedWith(featureOrdering),
                2,
            )
            val gp = GaussianProcessSurrogate(GaussianProcessConfig(hyperparameterTrials = 0)).fit(
                TrainingData(
                    baseline,
                    DenseVector.of(DoubleArray(size) { baseline[it, 0] * baseline[it, 0] + baseline[it, 1] }),
                    DenseVector.of(DoubleArray(size) { 1.0 }),
                    DenseVector.of(DoubleArray(size) { 0.001 }),
                ),
                Random(1),
            )
            val candidates = List(20) {
                val coordinates = List(
                    q,
                ) { listOf(random.nextDouble(), random.nextDouble()) }.sortedWith(featureOrdering)
                featureMatrix(coordinates, 2)
            }
            repeat(45) { iteration ->
                val fullValues = mutableListOf<Double>()
                val cachedValues = mutableListOf<Double>()
                var cached: JointGaussianSampler? = null
                fun full(): Long {
                    val start = System.nanoTime()
                    val normal = Random(77)
                    val z = DenseMatrix.zero(size + q, 256)
                    for (i in 0 until z.rows) {
                        for (j in 0 until z.cols step 2) {
                            z[i, j] = normal.nextNormal()
                            z[i, j + 1] = -z[i, j]
                        }
                    }
                    for (points in candidates) {
                        val joined = DenseMatrix.zero(size + q, 2)
                        for (i in 0 until joined.rows) {
                            for (j in 0..1) {
                                joined[i, j] =
                                    if (i < size) baseline[i, j] else points[i - size, j]
                            }
                        }
                        val moments = gp.predictJoint(joined)
                        val values = koblas.gemm(moments.covariance.factorCholesky().lowerFactor(), z)
                        for (i in 0 until values.rows) for (j in 0 until values.cols) values[i, j] += moments.means[i]
                        fullValues += improvement(size, q) { i, draw -> values[i, draw] }
                    }
                    return System.nanoTime() - start
                }
                fun cached(): Long {
                    val start = System.nanoTime()
                    val sampler = JointGaussianSampler(gp.jointPosterior(baseline), 256, 77).also { cached = it }
                    for (points in candidates) {
                        val values = sampler.draw(points)
                        cachedValues += improvement(size, q) { i, draw ->
                            if (i < size) sampler.baselineDraws[i, draw] else values[i - size, draw]
                        }
                    }
                    return System.nanoTime() - start
                }
                val fullNanos: Long
                val cachedNanos: Long
                if (iteration % 2 ==
                    0
                ) {
                    fullNanos = full()
                    cachedNanos = cached()
                } else {
                    cachedNanos = cached()
                    fullNanos = full()
                }
                val error = MaeLossStat()
                for (i in fullValues.indices) {
                    check(abs(fullValues[i] - cachedValues[i]) < 1e-8)
                    error.update(cachedValues[i], fullValues[i], 0L)
                }
                if (iteration >= 30) {
                    output.appendText(
                        "$size,$q,${iteration - 30},20,$fullNanos,$cachedNanos,${size + q},$size,$q," +
                            "${cached!!.baselineFactorizations},${cached!!.candidateFactorizations},${error.read(
                                0L,
                            ).mean}\n",
                    )
                }
            }
        }
    }
}
private fun improvement(baseline: Int, q: Int, value: (Int, Int) -> Double): Double {
    val stat = MeanStat()
    repeat(256) { draw ->
        val best = (0 until baseline).maxOf { value(it, draw) }
        val candidate = (baseline until baseline + q).maxOf { value(it, draw) }
        stat.update(maxOf(0.0, candidate - best), 0L)
    }
    return stat.read(0L).mean
}
