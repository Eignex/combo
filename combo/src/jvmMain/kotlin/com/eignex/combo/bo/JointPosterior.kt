package com.eignex.combo.bo

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector

/** Candidate latent moments and their covariance with a previously bound baseline. */
class JointPosteriorPrediction(
    /** Candidate means and covariance in the posterior's response coordinates. */
    val candidates: JointPredictiveMoments,
    /** Baseline-by-candidate latent covariance, with no observation noise. */
    val crossCovariance: DenseMatrix,
) {
    init {
        require(crossCovariance.cols == candidates.means.size)
    }
}

/**
 * Immutable posterior bound to fixed feature rows. This is a prediction cache, not conditioning on
 * invented baseline outcomes. Implementations capture baseline inputs and retain cross-point covariance.
 * Returned moments are read-only by contract; callers must copy before modifying them.
 */
interface JointPosterior {
    /** Fixed baseline latent moments. */
    val baseline: JointPredictiveMoments

    /** Candidate moments and baseline cross-covariance without re-evaluating baseline-only work when supported. */
    fun predict(features: DenseMatrix): JointPosteriorPrediction
}

/** Optional optimized fixed-baseline prediction capability. Decorators must opt in explicitly. */
interface JointPosteriorSurrogate : FittedSurrogate {
    /** Capture a baseline and cache reusable model-specific joint prediction work. */
    fun jointPosterior(baseline: DenseMatrix): JointPosterior?
}

/** Optional optimized joint prediction in Gaussian log-response coordinates. */
interface LogJointPosteriorSurrogate :
    LogResponseSurrogate,
    JointPosteriorSurrogate {
    /** Capture a baseline before applying the lognormal response mapping. */
    fun logJointPosterior(baseline: DenseMatrix): JointPosterior?
}

/**
 * Bind joint predictions, using an explicitly advertised cache capability when available.
 * Other models and decorators fall back to their own joint predictions, preserving overrides.
 */
fun FittedSurrogate.jointPosterior(baseline: DenseMatrix): JointPosterior? =
    if (this is JointPosteriorSurrogate) jointPosterior(baseline) else bindJointPosterior(baseline, ::predictJoint)

/** Bind Gaussian log-response predictions, preserving model-specific caches only when explicitly supported. */
fun LogResponseSurrogate.logJointPosterior(baseline: DenseMatrix): JointPosterior? =
    if (this is LogJointPosteriorSurrogate) {
        logJointPosterior(
            baseline,
        )
    } else {
        bindJointPosterior(baseline, ::predictLogJoint)
    }

internal fun bindJointPosterior(
    features: DenseMatrix,
    predict: (DenseMatrix) -> JointPredictiveMoments?,
): JointPosterior? {
    val fixed = DenseMatrix.wrap(features.rows, features.cols, features.values.copyOf())
    val moments = predict(fixed) ?: return null
    return object : JointPosterior {
        override val baseline = moments
        override fun predict(features: DenseMatrix): JointPosteriorPrediction {
            require(features.cols == fixed.cols)
            val joined = DenseMatrix.zero(fixed.rows + features.rows, fixed.cols)
            for (i in 0 until joined.rows) {
                for (j in 0 until joined.cols) {
                    joined[i, j] = if (i < fixed.rows) fixed[i, j] else features[i - fixed.rows, j]
                }
            }
            val all = requireNotNull(predict(joined))
            val covariance = DenseMatrix.zero(features.rows)
            val cross = DenseMatrix.zero(fixed.rows, features.rows)
            for (i in 0 until features.rows) {
                for (j in 0 until features.rows) covariance[i, j] = all.covariance[fixed.rows + i, fixed.rows + j]
                for (j in 0 until fixed.rows) cross[j, i] = all.covariance[j, fixed.rows + i]
            }
            return JointPosteriorPrediction(
                JointPredictiveMoments(
                    DenseVector.of(DoubleArray(features.rows) { all.means[fixed.rows + it] }),
                    covariance,
                ),
                cross,
            )
        }
    }
}
