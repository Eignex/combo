package com.eignex.combo.bo.model

import kotlin.math.ln
import kotlin.math.sqrt

/** Constant mean estimation when GaussianProcessConfig.mean is unspecified. */
enum class GaussianMeanEstimation {
    /** Average the original training responses once, independently of covariance. */
    Empirical,

    /** Profile the common mean by generalized least squares at each covariance candidate. */
    CovarianceWeighted,
}

/** Bounded hyperparameter search under the same total covariance-candidate budget. */
enum class GaussianHyperparameterSearch {
    /** Refine one initial point by coordinate search. */
    Coordinate,

    /** Split the budget over seeded log-space starts, retaining the best candidate across all starts. */
    Multistart,
}

/**
 * Independent Gaussian penalties on log covariance parameters, with constants omitted.
 * This is a density in log-parameter coordinates, without a Jacobian for a density in original coordinates.
 * Inputs should be normalized and responses standardized. Parameters remain point estimates;
 * posterior predictions do not integrate hyperparameter uncertainty.
 */
data class GaussianProcessRegularization(
    /** Length-scale center; null uses 0.5 times the square root of the number of coordinates. */
    val lengthScale: Double? = null,
    /** Standard deviation of each log length scale around its log center. */
    val lengthScaleLogDeviation: Double = 1.0,
    /** Signal variance center in training-response units. */
    val signalVariance: Double = 1.0,
    /** Standard deviation of log signal variance. */
    val signalLogDeviation: Double = 2.0,
    /** Unknown-noise variance center; null retains the fit's initial unknown-noise variance. */
    val noiseVariance: Double? = null,
    /** Standard deviation of log unknown-noise variance; known row variances are never penalized. */
    val noiseLogDeviation: Double = 2.0,
) {
    init {
        require(lengthScale == null || (lengthScale > 0.0 && lengthScale.isFinite()))
        require(signalVariance > 0.0 && signalVariance.isFinite())
        require(noiseVariance == null || (noiseVariance > 0.0 && noiseVariance.isFinite()))
        require(
            listOf(lengthScaleLogDeviation, signalLogDeviation, noiseLogDeviation).all {
                it > 0.0 && it.isFinite()
            },
        )
    }

    internal fun penalty(parameters: GaussianProcessParameters, noiseCenter: Double?): Double {
        val center = lengthScale ?: (0.5 * sqrt(maxOf(1, parameters.lengthScales.size).toDouble()))
        fun square(value: Double, target: Double, deviation: Double): Double {
            val residual = (ln(value) - ln(target)) / deviation
            return residual * residual
        }
        val lengthPenalty = parameters.lengthScales.sumOf { square(it, center, lengthScaleLogDeviation) }
        val signalPenalty = square(parameters.signalVariance, signalVariance, signalLogDeviation)
        val noisePenalty = noiseCenter?.let {
            square(parameters.noiseVariance, noiseVariance ?: it, noiseLogDeviation)
        } ?: 0.0
        return -0.5 * (lengthPenalty + signalPenalty + noisePenalty)
    }
}

/**
 * Optional GP fitting policies. Defaults preserve empirical-mean, unregularized coordinate fitting.
 * GaussianProcessConfig.hyperparameterTrials counts all candidates after the initial fit, including
 * restart points. A fixed configured mean overrides meanEstimation. Conditioning freezes the fitted mean.
 */
data class GaussianProcessFitting(
    /** Policy for estimating an unspecified constant mean. */
    val meanEstimation: GaussianMeanEstimation = GaussianMeanEstimation.Empirical,
    /** Optional penalty added to log marginal likelihood for candidate selection. */
    val regularization: GaussianProcessRegularization? = null,
    /** Covariance search strategy. */
    val search: GaussianHyperparameterSearch = GaussianHyperparameterSearch.Coordinate,
    /** Maximum number of starts, including the configured initial point. */
    val starts: Int = 4,
) {
    init {
        require(starts > 0)
    }
}
