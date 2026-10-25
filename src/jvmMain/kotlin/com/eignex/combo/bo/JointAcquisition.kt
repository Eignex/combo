package com.eignex.combo.bo

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.koblas
import com.eignex.kumulant.math.NotPositiveDefinite
import com.eignex.kumulant.math.factorCholesky
import com.eignex.kumulant.math.nextNormal
import com.eignex.kumulant.stat.summary.MeanStat
import kotlin.math.exp
import kotlin.random.Random

/** A fixed acquisition objective scoring an entire set of feature rows together. */
fun interface JointAcquisition {
    /** Expected incremental utility of this candidate set, always maximized. */
    fun evaluate(features: DenseMatrix): Double
}

/** Frozen model, outcome, and same-context completed/pending coordinates for joint acquisition. */
class JointAcquisitionContext(
    /** Immutable fitted model exposing joint moments. */
    val model: FittedSurrogate,
    /** Fixed improvement baseline in original response units, used by noiseless EI. */
    val incumbent: Double?,
    /** Objective orientation and response policy. */
    val outcome: OutcomeSpec,
    /** Previously evaluated coordinates; noisy EI samples their latent values jointly with candidates. */
    val baseline: DenseMatrix,
    /** Outstanding coordinates whose latent values enter the incremental improvement baseline. */
    val pending: DenseMatrix = DenseMatrix.zero(0, baseline.cols),
)

/** Bind a joint acquisition to a frozen posterior and fixed Monte Carlo random stream. */
fun interface JointAcquisitionFactory {
    /** Randomness is captured here, never redrawn for successive evaluations of the same set. */
    fun create(context: JointAcquisitionContext, random: Random): JointAcquisition
}

/**
 * Monte Carlo q-EI using a Gaussian approximation to joint latent moments and a fixed incumbent.
 * Pending points raise the sampled improvement threshold; their own improvement is not counted.
 * Log-response models are sampled in Gaussian log coordinates and scored in original units.
 * Reuses fixed antithetic normal samples, canonicalizes row order, and deduplicates coordinates.
 * Near-singular covariance can receive relative numerical jitter up to 1e-8; this is not observation noise.
 * Each bound objective caches normal samples by dimension and is single-threaded.
 *
 * @property samples Even number of Monte Carlo draws per evaluated set.
 * @property margin Required improvement in original response units.
 */
class JointExpectedImprovement(val samples: Int = 256, val margin: Double = 0.0) : JointAcquisitionFactory {
    init {
        validateJointSettings(samples, margin)
    }
    override fun create(context: JointAcquisitionContext, random: Random): JointAcquisition {
        require(context.incumbent?.isFinite() == true) { "q-EI requires a finite incumbent" }
        return jointImprovement(context, random, samples, margin, noisy = false)
    }
}

/**
 * Monte Carlo q-NEI sampling candidates and completed/pending baselines from one joint latent Gaussian.
 * No observed best response is treated as certain. Samples exclude observation noise, and pending labels
 * never enter recorded history. Shares [JointExpectedImprovement]'s sampling and numerical conventions.
 * Baselines are not pruned or root-cached: each set factors its joint covariance, so large histories cost more.
 *
 * @property samples Even number of Monte Carlo draws per evaluated set.
 * @property margin Required improvement in original response units.
 */
class JointNoisyExpectedImprovement(val samples: Int = 256, val margin: Double = 0.0) : JointAcquisitionFactory {
    init {
        validateJointSettings(samples, margin)
    }
    override fun create(context: JointAcquisitionContext, random: Random): JointAcquisition {
        require(context.baseline.rows > 0) { "q-NEI requires completed baseline coordinates" }
        return jointImprovement(context, random, samples, margin, noisy = true)
    }
}

private fun validateJointSettings(samples: Int, margin: Double) {
    require(samples > 0 && samples % 2 == 0)
    require(margin >= 0.0 && margin.isFinite())
}

private fun featureKeys(features: DenseMatrix): List<List<Double>> {
    require(features.values.all { it.isFinite() })
    return List(features.rows) { i -> List(features.cols) { j -> features[i, j].let { if (it == 0.0) 0.0 else it } } }
}

private fun jointImprovement(
    context: JointAcquisitionContext,
    random: Random,
    samples: Int,
    margin: Double,
    noisy: Boolean,
): JointAcquisition {
    val dimensions = context.baseline.cols
    require(context.pending.cols == dimensions)
    val baseline = (if (noisy) featureKeys(context.baseline) else emptyList()) + featureKeys(context.pending)
    val model = context.model
    val logarithmic = model is LogResponseSurrogate
    fun moments(features: DenseMatrix): JointPredictiveMoments = requireNotNull(
        if (model is LogResponseSurrogate) model.predictLogJoint(features) else model.predictJoint(features),
    ) { "joint EI requires joint Gaussian moments; marginal-only models are unsupported" }
    moments(DenseMatrix.zero(0, dimensions))
    val seed = random.nextLong()
    val normals = mutableMapOf<Int, DenseMatrix>()
    return JointAcquisition { features ->
        require(features.cols == dimensions)
        val candidates = featureKeys(features).distinct()
        if (candidates.isEmpty()) return@JointAcquisition 0.0
        val ordering = Comparator<List<Double>> { a, b ->
            var order = 0
            for (i in a.indices) {
                order = a[i].compareTo(b[i])
                if (order != 0) break
            }
            order
        }
        // Keep baseline coordinates first so their Cholesky block and base draws stay fixed as candidates change.
        val fixed = baseline.distinct().sortedWith(ordering)
        val keys = fixed + (candidates - fixed.toSet()).sortedWith(ordering)
        val indices = keys.withIndex().associate { it.value to it.index }
        val baselineRows = baseline.distinct().map { indices.getValue(it) }
        val candidateRows = candidates.map { indices.getValue(it) }
        val queries = DenseMatrix.zero(keys.size, dimensions)
        for (i in keys.indices) for (j in 0 until dimensions) queries[i, j] = keys[i][j]
        val joint = moments(queries)
        require(joint.means.size == keys.size && joint.covariance.rows == keys.size)
        require(joint.means.toDoubleArray().all { it.isFinite() } && joint.covariance.values.all { it.isFinite() })
        require(keys.indices.all { joint.covariance[it, it] >= 0.0 })
        val active = keys.indices.filter { joint.covariance[it, it] > 0.0 }
        val positions = IntArray(keys.size) { -1 }
        active.forEachIndexed { position, row -> positions[row] = position }
        val draws = if (active.isEmpty()) {
            null
        } else {
            val covariance = DenseMatrix.zero(active.size)
            for (i in active.indices) {
                for (j in active.indices) {
                    covariance[i, j] =
                        joint.covariance[active[i], active[j]]
                }
            }
            val lower = jointRoot(covariance)
            val z = normals.getOrPut(active.size) {
                val generator = Random(seed)
                DenseMatrix.zero(active.size, samples).also { values ->
                    for (row in active.indices) {
                        for (column in 0 until samples step 2) {
                            val normal = generator.nextNormal()
                            values[row, column] = normal
                            values[row, column + 1] = -normal
                        }
                    }
                }
            }
            koblas.gemm(lower, z)
        }
        fun value(row: Int, draw: Int): Double {
            val sample = joint.means[row] + if (positions[row] < 0) 0.0 else draws!![positions[row], draw]
            return context.outcome.score(if (logarithmic) exp(sample) else sample)
        }
        val utility = MeanStat()
        repeat(samples) { draw ->
            var threshold = if (noisy) Double.NEGATIVE_INFINITY else context.outcome.score(context.incumbent!!)
            for (row in baselineRows) threshold = maxOf(threshold, value(row, draw))
            val best = candidateRows.maxOf { value(it, draw) }
            utility.update(maxOf(0.0, best - threshold - margin), 0L)
        }
        utility.read(0L).mean.also { require(it.isFinite()) { "nonfinite joint EI; rescale responses" } }
    }
}

private fun jointRoot(covariance: DenseMatrix): DenseMatrix {
    try {
        return covariance.factorCholesky().lowerFactor()
    } catch (_: NotPositiveDefinite) {
        // Joint posteriors can be singular even after exact duplicate coordinates are removed.
    }
    val scale = (0 until covariance.rows).maxOf { covariance[it, it] }
    var jitter = scale * 1e-12
    var failure: NotPositiveDefinite? = null
    repeat(5) {
        val stabilized = DenseMatrix.wrap(covariance.rows, covariance.cols, covariance.values.copyOf())
        for (i in 0 until covariance.rows) stabilized[i, i] += jitter
        try {
            return stabilized.factorCholesky().lowerFactor()
        } catch (error: NotPositiveDefinite) {
            failure = error
            jitter *= 10.0
        }
    }
    throw requireNotNull(failure)
}
