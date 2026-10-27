package com.eignex.combo.bo

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.koblas
import com.eignex.kumulant.math.CholeskyFactor
import com.eignex.kumulant.math.NotPositiveDefinite
import com.eignex.kumulant.math.SobolSequence
import com.eignex.kumulant.math.factorCholesky
import com.eignex.kumulant.math.nextNormal
import kotlin.random.Random

internal fun featureKeys(features: DenseMatrix): List<List<Double>> {
    require(features.values.all { it.isFinite() })
    return List(features.rows) { i -> List(features.cols) { j -> features[i, j].let { if (it == 0.0) 0.0 else it } } }
}
internal val featureOrdering = Comparator<List<Double>> { a, b ->
    var order = 0
    for (i in a.indices) {
        order = a[i].compareTo(b[i])
        if (order != 0) break
    }
    order
}
internal fun featureMatrix(rows: List<List<Double>>, dimensions: Int): DenseMatrix = DenseMatrix.zero(
    rows.size,
    dimensions,
).also {
    for (i in rows.indices) for (j in 0 until dimensions) it[i, j] = rows[i][j]
}

/** BO posterior sampling cache: model-specific baseline prediction work and its factor are frozen once. */
internal class JointGaussianSampler(
    private val posterior: JointPosterior,
    private val samples: Int,
    private val seed: Long,
    private val sampling: JointNormalSampling = JointNormalSampling.Antithetic,
) {
    private val baseline = posterior.baseline
    private val active = baseline.means.toDoubleArray().indices.filter { baseline.covariance[it, it] > 0.0 }
    private val factor: CholeskyFactor?
    private val generator = Random(seed)
    private val normals = mutableListOf<DoubleArray>()
    private val baselineNormals: DenseMatrix
    val baselineDraws: DenseMatrix
    var baselineFactorizations = 0
        private set
    var candidateFactorizations = 0
        private set

    init {
        validate(baseline)
        val covariance = DenseMatrix.zero(active.size)
        for (i in active.indices) for (j in active.indices) covariance[i, j] = baseline.covariance[active[i], active[j]]
        factor = if (active.isEmpty()) null else jointFactor(covariance).also { baselineFactorizations++ }
        baselineNormals = normalRows(0, active.size)
        baselineDraws = DenseMatrix.zero(baseline.means.size, samples)
        val deviations = factor?.let { koblas.gemm(it.lowerFactor(), baselineNormals) }
        val positions = active.withIndex().associate { it.value to it.index }
        for (i in 0 until baselineDraws.rows) {
            for (draw in 0 until samples) {
                baselineDraws[i, draw] = baseline.means[i] + (positions[i]?.let { deviations!![it, draw] } ?: 0.0)
            }
        }
    }

    private fun normalRows(start: Int, count: Int): DenseMatrix {
        if (sampling == JointNormalSampling.ScrambledSobol && normals.size < start + count) {
            require(start + count <= SobolSequence.MAX_DIMENSIONS) {
                "scrambled Sobol supports at most ${SobolSequence.MAX_DIMENSIONS} latent coordinates; use Antithetic"
            }
            val first = normals.size
            val width = start + count - first
            val sequence = SobolSequence(width, seed, dimensionOffset = first)
            val point = DoubleArray(width)
            val columns = List(width) { DoubleArray(samples) }
            repeat(samples) { draw ->
                sequence.nextNormal(point)
                for (coordinate in point.indices) columns[coordinate][draw] = point[coordinate]
            }
            normals.addAll(columns)
        }
        while (normals.size < start + count) {
            normals += DoubleArray(samples).also { values ->
                for (i in 0 until samples step 2) {
                    values[i] = generator.nextNormal()
                    values[i + 1] = -values[i]
                }
            }
        }
        return DenseMatrix.zero(count, samples).also { z ->
            for (i in 0 until count) for (draw in 0 until samples) z[i, draw] = normals[start + i][draw]
        }
    }

    fun draw(features: DenseMatrix): DenseMatrix {
        val prediction = posterior.predict(features)
        val moments = prediction.candidates
        validate(moments)
        require(moments.means.size == features.rows && prediction.crossCovariance.rows == baseline.means.size)
        require(prediction.crossCovariance.values.all { it.isFinite() })
        val cross = DenseMatrix.zero(active.size, features.rows)
        for (i in active.indices) {
            for (j in 0 until features.rows) {
                cross[i, j] =
                    prediction.crossCovariance[active[i], j]
            }
        }
        val solved = factor?.solveLower(cross) ?: cross
        val conditional = DenseMatrix.wrap(features.rows, features.rows, moments.covariance.values.copyOf())
        koblas.gemm(-1.0, solved, true, solved, false, 1.0, conditional)
        val scale = moments.means.toDoubleArray().indices.maxOfOrNull { moments.covariance[it, it] } ?: 0.0
        for (i in 0 until conditional.rows) {
            if (conditional[i, i] < -scale * 1e-10) {
                throw NotPositiveDefinite(i, conditional[i, i], "negative conditional GP variance")
            }
            conditional[i, i] = maxOf(0.0, conditional[i, i])
        }
        val stochastic = (0 until conditional.rows).filter { conditional[it, it] > 0.0 }
        val covariance = DenseMatrix.zero(stochastic.size)
        for (i in stochastic.indices) {
            for (j in stochastic.indices) {
                covariance[i, j] =
                    conditional[stochastic[i], stochastic[j]]
            }
        }
        val independent = if (stochastic.isEmpty()) {
            null
        } else {
            candidateFactorizations++
            koblas.gemm(jointFactor(covariance).lowerFactor(), normalRows(active.size, stochastic.size))
        }
        val shared = DenseMatrix.zero(features.rows, samples)
        koblas.gemm(1.0, solved, true, baselineNormals, false, 0.0, shared)
        val positions = stochastic.withIndex().associate { it.value to it.index }
        return DenseMatrix.zero(features.rows, samples).also { draws ->
            for (i in 0 until features.rows) {
                for (draw in 0 until samples) {
                    draws[i, draw] =
                        moments.means[i] + shared[i, draw] + (positions[i]?.let { independent!![it, draw] } ?: 0.0)
                }
            }
        }
    }
}

private fun validate(moments: JointPredictiveMoments) {
    require(moments.means.toDoubleArray().all { it.isFinite() } && moments.covariance.values.all { it.isFinite() })
    require((0 until moments.means.size).all { moments.covariance[it, it] >= 0.0 })
}

private fun jointFactor(covariance: DenseMatrix): CholeskyFactor {
    try {
        return covariance.factorCholesky()
    } catch (_: NotPositiveDefinite) {
        // A joint latent posterior may be singular even after duplicate feature rows are removed.
    }
    val scale = (0 until covariance.rows).maxOf { covariance[it, it] }
    var jitter = scale * 1e-12
    var failure: NotPositiveDefinite? = null
    repeat(5) {
        val stabilized = DenseMatrix.wrap(covariance.rows, covariance.cols, covariance.values.copyOf())
        for (i in 0 until covariance.rows) stabilized[i, i] += jitter
        try {
            return stabilized.factorCholesky()
        } catch (
            error: NotPositiveDefinite,
        ) {
            failure = error
            jitter *= 10.0
        }
    }
    throw requireNotNull(failure)
}
