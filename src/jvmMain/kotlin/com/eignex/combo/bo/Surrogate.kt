package com.eignex.combo.bo

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.Vector
import com.eignex.kumulant.stat.regression.tree.BatchForestConfig
import com.eignex.kumulant.stat.regression.tree.BatchRandomForest
import com.eignex.kumulant.stat.regression.tree.RegressionObservation
import com.eignex.kumulant.stat.regression.tree.predictDistribution
import kotlin.random.Random

/** Numeric training data, with one observation per matrix row. Noise variances are separate from weights. */
class TrainingData(
    /** Feature rows. Fitters must capture any storage they retain. */
    val features: DenseMatrix,
    /** Responses in the fitter's outcome coordinates. */
    val responses: Vector,
    /** Positive statistical weights. */
    val weights: Vector,
    /** Optional known observation noise, NaN where unspecified. */
    val noiseVariances: Vector? = null,
) {
    init {
        require(features.rows == responses.size && responses.size == weights.size)
        require(noiseVariances == null || noiseVariances.size == responses.size)
        require(features.values.all { it.isFinite() } && responses.toDoubleArray().all { it.isFinite() })
        require(weights.toDoubleArray().all { it > 0.0 && it.isFinite() })
        require(
            noiseVariances == null || noiseVariances.toDoubleArray().all { it.isNaN() || (it >= 0.0 && it.isFinite()) },
        )
    }
}

/** Meaning of the uncertainty exposed by a fitted model. */
enum class UncertaintyKind {
    /** Dispersion across randomized ensemble members, without a calibration guarantee. */
    EnsembleDisagreement,

    /** A model's probabilistic uncertainty about its latent function. */
    LatentPosterior,
}

/** Batch latent-function moments, in original reward units. Observation noise is not added to variance. */
class PredictiveMoments(
    /** One latent mean per query row. */
    val means: Vector,
    /** One nonnegative latent variance per query row. */
    val variances: Vector,
) {
    init {
        require(means.size == variances.size)
    }
}

/** A fixed function draw. Repeated and reordered evaluations must describe the same sampled function. */
fun interface SampledFunction {
    /** Score all feature rows using this draw. */
    fun evaluate(features: DenseMatrix): Vector
}

/** Immutable fitted surrogate, independent of the trial lifecycle and acquisition rule. */
interface FittedSurrogate {
    /** Statistical interpretation of predictive variance. */
    val uncertaintyKind: UncertaintyKind

    /** Evaluate latent moments for all query rows. */
    fun predict(features: DenseMatrix): PredictiveMoments

    /** Draw a function once, preserving cross-point dependence throughout its evaluations. */
    fun drawFunction(random: Random): SampledFunction
}

/** Optional capability for models that can condition on temporary observations without a full refit. */
interface ConditionableSurrogate : FittedSurrogate {
    /** Return an independent conditioned model; never change this model or recorded history. */
    fun condition(temporary: TrainingData): FittedSurrogate
}

/** Stateless full-history fitter. Implementations must not mutate inputs or previous fitted models. */
fun interface Surrogate {
    /** Fit an independent model using the supplied reproducible random stream. */
    fun fit(data: TrainingData, random: Random): FittedSurrogate
}

/**
 * Batch CART forest adapter; between-tree variance is explicitly ensemble disagreement.
 * Known observation noise is retained in the dataset but is not modeled by this forest fitter.
 */
class ForestSurrogate(
    /** Kumulant's numeric forest configuration. */
    val config: BatchForestConfig = BatchForestConfig(),
) : Surrogate {
    override fun fit(data: TrainingData, random: Random): FittedSurrogate {
        val featureSize = data.features.cols
        val rows = List(data.features.rows) { i ->
            RegressionObservation(data.features.rowVector(i), data.responses[i], data.weights[i])
        }
        val forest = BatchRandomForest(data.features.cols, config).fit(rows, random)
        return object : FittedSurrogate {
            override val uncertaintyKind = UncertaintyKind.EnsembleDisagreement
            override fun predict(features: DenseMatrix): PredictiveMoments {
                require(features.cols == featureSize)
                val moments = List(features.rows) { forest.predictDistribution(features.rowVector(it)) }
                return PredictiveMoments(
                    DenseVector.of(DoubleArray(features.rows) { moments[it].mean }),
                    DenseVector.of(DoubleArray(features.rows) { moments[it].variance }),
                )
            }
            override fun drawFunction(random: Random): SampledFunction {
                val tree = forest.trees[random.nextInt(forest.trees.size)]
                return SampledFunction { features ->
                    require(features.cols == featureSize)
                    DenseVector.of(DoubleArray(features.rows) { tree.predict(features.rowVector(it)) })
                }
            }
        }
    }
}

internal fun DenseMatrix.rowVector(row: Int): Vector = DenseVector.of(DoubleArray(cols) { this[row, it] })
internal fun featureRows(rows: List<Vector>, size: Int): DenseMatrix = if (rows.isEmpty()) {
    DenseMatrix.zero(0, size)
} else {
    DenseMatrix.ofRows(rows.map { it.toDoubleArray() }.toTypedArray())
}
