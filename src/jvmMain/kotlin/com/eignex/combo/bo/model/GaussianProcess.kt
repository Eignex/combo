package com.eignex.combo.bo.model

import com.eignex.combo.bo.JointPosterior
import com.eignex.combo.bo.JointPosteriorPrediction
import com.eignex.combo.bo.JointPredictiveMoments
import com.eignex.combo.bo.PredictiveMoments
import com.eignex.combo.bo.SampledFunction
import com.eignex.combo.bo.TrainingData
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.koblas
import com.eignex.kumulant.math.CholeskyFactor
import com.eignex.kumulant.math.NotPositiveDefinite
import com.eignex.kumulant.math.factorCholesky
import com.eignex.kumulant.math.nextNormal
import com.eignex.kumulant.stat.summary.estimateMean
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/** Stationary covariance families for numeric search coordinates. */
enum class GaussianKernel {
    /** Matérn covariance with smoothness five halves. */
    Matern52,

    /** Squared exponential covariance. */
    Rbf,
}

/** Exact full-history GP configuration; variance and mean parameters use training-response units. */
data class GaussianProcessConfig(
    /** Covariance family; inputs should use meaningful numeric geometry such as SearchTransform. */
    val kernel: GaussianKernel = GaussianKernel.Matern52,
    /** Initial common length scale when per-feature scales are absent. */
    val lengthScale: Double = 0.25,
    /** Optional initial length scale per coordinate, copied on fit. */
    val lengthScales: List<Double>? = null,
    /** Initial latent function variance. */
    val signalVariance: Double = 1.0,
    /** Observation variance for rows without a supplied noise variance; zero permits exact observations. */
    val noiseVariance: Double = 1e-6,
    /** Fixed constant prior mean; null estimates one mean from the original training responses. */
    val mean: Double? = null,
    /** Number of bounded coordinate-search likelihood candidates; zero fixes supplied parameters. */
    val hyperparameterTrials: Int = 24,
    /** Include unknown observation noise in likelihood search; supplied row variances remain fixed. */
    val learnNoise: Boolean = false,
    /** Initial numerical diagonal stabilization relative to covariance scale, separate from noise. */
    val jitter: Double = 1e-10,
    /** Maximum factorization attempts, increasing numerical jitter tenfold after each failed pivot. */
    val factorizationAttempts: Int = 6,
) {
    init {
        require(lengthScale > 0.0 && lengthScale.isFinite())
        require(lengthScales == null || lengthScales.all { it > 0.0 && it.isFinite() })
        require(signalVariance > 0.0 && signalVariance.isFinite())
        require(noiseVariance >= 0.0 && noiseVariance.isFinite())
        require(mean == null || mean.isFinite())
        require(hyperparameterTrials >= 0 && jitter > 0.0 && jitter.isFinite() && factorizationAttempts > 0)
    }
}

/** Fixed fitted covariance and mean parameters, conditional on which predictions form a GP posterior. */
data class GaussianProcessParameters(
    /** Length scale per numeric coordinate. */
    val lengthScales: List<Double>,
    /** Latent function variance. */
    val signalVariance: Double,
    /** Observation variance used only where a row has no supplied variance. */
    val noiseVariance: Double,
    /** Constant mean frozen for conditioning. */
    val mean: Double,
)

internal class GaussianProcessModel private constructor(
    private val data: TrainingData,
    private val config: GaussianProcessConfig,
    val parameters: GaussianProcessParameters,
    private val factor: CholeskyFactor,
    private val alpha: DoubleArray,
    val numericalJitter: Double,
    val logMarginalLikelihood: Double,
) {
    private val size get() = data.features.rows
    private val dimensions get() = data.features.cols

    private fun validate(features: DenseMatrix) {
        require(features.cols == dimensions) { "GP feature dimension mismatch" }
        require(features.values.all { it.isFinite() }) { "GP features must be finite" }
    }

    private fun cross(features: DenseMatrix): DenseMatrix = covariance(
        data.features,
        features,
        config.kernel,
        parameters,
    )

    fun predict(features: DenseMatrix): PredictiveMoments {
        validate(features)
        val covariance = cross(features)
        val means = koblas.gemv(covariance, alpha, transpose = true)
        for (i in means.indices) means[i] += parameters.mean
        val solved = factor.solveLower(covariance)
        val variances = DoubleArray(features.rows) { column ->
            val square = koblas.vectorKernels.dot(solved.values, column * size, solved.values, column * size, size)
            maxOf(0.0, parameters.signalVariance - square)
        }
        return PredictiveMoments(DenseVector.of(means), DenseVector.of(variances))
    }

    fun predictJoint(features: DenseMatrix): JointPredictiveMoments = jointFeatures(features).first

    private fun jointFeatures(features: DenseMatrix): Pair<JointPredictiveMoments, DenseMatrix> {
        validate(features)
        val cross = cross(features)
        val means = koblas.gemv(cross, alpha, transpose = true)
        for (i in means.indices) means[i] += parameters.mean
        val solved = factor.solveLower(cross)
        val result = covariance(features, features, config.kernel, parameters)
        koblas.gemm(-1.0, solved, true, solved, false, 1.0, result)
        for (i in 0 until result.rows) result[i, i] = maxOf(0.0, result[i, i])
        return JointPredictiveMoments(DenseVector.of(means), result) to solved
    }

    fun jointPosterior(baseline: DenseMatrix): JointPosterior {
        val fixed = DenseMatrix.wrap(baseline.rows, baseline.cols, baseline.values.copyOf())
        val (moments, baselineSolved) = jointFeatures(fixed)
        return object : JointPosterior {
            override val baseline = moments
            override fun predict(features: DenseMatrix): JointPosteriorPrediction {
                val (candidates, solved) = jointFeatures(features)
                val cross = covariance(fixed, features, config.kernel, parameters)
                koblas.gemm(-1.0, baselineSolved, true, solved, false, 1.0, cross)
                return JointPosteriorPrediction(candidates, cross)
            }
        }
    }

    fun condition(temporary: TrainingData): GaussianProcessModel {
        validate(temporary.features)
        validateWeights(temporary)
        if (temporary.features.rows == 0) return this
        val joined = DenseMatrix.zero(size + temporary.features.rows, dimensions)
        for (i in 0 until joined.rows) {
            for (j in 0 until dimensions) {
                joined[i, j] = if (i < size) data.features[i, j] else temporary.features[i - size, j]
            }
        }
        val all = TrainingData(
            joined,
            DenseVector.of(data.responses.toDoubleArray() + temporary.responses.toDoubleArray()),
            DenseVector.of(DoubleArray(joined.rows) { 1.0 }),
            DenseVector.of(
                DoubleArray(joined.rows) {
                    if (it < size) {
                        data.noiseVariances?.get(it) ?: Double.NaN
                    } else {
                        temporary.noiseVariances?.get(it - size) ?: Double.NaN
                    }
                },
            ),
        )
        return build(all, config, parameters)
    }

    fun drawFunction(random: Random): SampledFunction {
        val generator = Random(random.nextLong())
        val cache = mutableMapOf<List<Double>, Double>()
        var conditioned = this
        return SampledFunction { features ->
            validate(features)
            val keys = List(features.rows) { row ->
                List(dimensions) { column ->
                    features[row, column].let { if (it == 0.0) 0.0 else it }
                }
            }
            val missing = keys.distinct().filter { it !in cache }
            if (missing.isNotEmpty()) {
                val queries = DenseMatrix.zero(missing.size, dimensions)
                for (i in missing.indices) for (j in 0 until dimensions) queries[i, j] = missing[i][j]
                val joint = conditioned.predictJoint(queries)
                val (sampling, _) = stabilized(joint.covariance, config, parameters.signalVariance)
                val deviations = koblas.gemv(
                    sampling.lowerFactor(),
                    DoubleArray(missing.size) { generator.nextNormal() },
                )
                val values = DoubleArray(missing.size) { joint.means[it] + deviations[it] }
                missing.forEachIndexed { index, key -> cache[key] = values[index] }
                conditioned = conditioned.condition(
                    TrainingData(
                        queries,
                        DenseVector.of(values),
                        DenseVector.of(DoubleArray(values.size) { 1.0 }),
                        DenseVector.zero(values.size),
                    ),
                )
            }
            DenseVector.of(DoubleArray(keys.size) { cache.getValue(keys[it]) })
        }
    }

    companion object {
        fun fit(input: TrainingData, config: GaussianProcessConfig): GaussianProcessModel {
            validateWeights(input)
            require(config.lengthScales == null || config.lengthScales.size == input.features.cols)
            val data = TrainingData(
                DenseMatrix.wrap(input.features.rows, input.features.cols, input.features.values.copyOf()),
                DenseVector.of(input.responses.toDoubleArray()),
                DenseVector.of(input.weights.toDoubleArray()),
                input.noiseVariances?.let { DenseVector.of(it.toDoubleArray()) },
            )
            val estimate = if (data.responses.size > 0) estimateMean(data.responses) else null
            val learnNoise = config.learnNoise && (0 until data.responses.size).any {
                data.noiseVariances?.get(it)?.isNaN() ?: true
            }
            val initial = GaussianProcessParameters(
                config.lengthScales?.toList() ?: List(input.features.cols) { config.lengthScale },
                config.signalVariance,
                if (learnNoise) {
                    maxOf(
                        config.noiseVariance,
                        (estimate?.responseVariance ?: config.signalVariance) * 0.01,
                        1e-12,
                    )
                } else {
                    config.noiseVariance
                },
                config.mean ?: estimate?.mean ?: 0.0,
            )
            var best = build(data, config, initial)
            if (data.features.rows == 0) return best
            val coordinates = input.features.cols + 1 + if (learnNoise) 1 else 0
            var origin = initial
            repeat(config.hyperparameterTrials) { trial ->
                if (trial % 2 == 0) origin = best.parameters
                val coordinate = (trial / 2) % coordinates
                val cycle = trial / (2 * coordinates)
                val step = ln(2.0) / (1 + cycle / 2) * if (trial % 2 == 0) -1.0 else 1.0
                val proposal = perturb(origin, coordinate, step)
                val candidate = build(data, config, proposal)
                if (candidate.logMarginalLikelihood > best.logMarginalLikelihood) best = candidate
            }
            return best
        }

        private fun validateWeights(data: TrainingData) {
            require(data.weights.toDoubleArray().all { it == 1.0 }) {
                "GP requires unit statistical weights; supply noise variances or independent replicate rows explicitly"
            }
        }

        private fun perturb(
            parameters: GaussianProcessParameters,
            coordinate: Int,
            step: Double,
        ): GaussianProcessParameters {
            val scales = parameters.lengthScales.toMutableList()
            if (coordinate < scales.size) {
                scales[coordinate] = exp((ln(scales[coordinate]) + step).coerceIn(ln(1e-4), ln(1e4)))
                return parameters.copy(lengthScales = scales)
            }
            return if (coordinate == scales.size) {
                parameters.copy(
                    signalVariance = exp((ln(parameters.signalVariance) + step).coerceIn(ln(1e-12), ln(1e12))),
                )
            } else {
                parameters.copy(
                    noiseVariance = exp((ln(parameters.noiseVariance) + step).coerceIn(ln(1e-12), ln(1e12))),
                )
            }
        }

        private fun build(
            data: TrainingData,
            config: GaussianProcessConfig,
            parameters: GaussianProcessParameters,
        ): GaussianProcessModel {
            val matrix = covariance(data.features, data.features, config.kernel, parameters)
            for (i in 0 until matrix.rows) {
                matrix[i, i] += data.noiseVariances?.get(
                    i,
                )?.takeUnless { it.isNaN() } ?: parameters.noiseVariance
            }
            val (factor, jitter) = stabilized(matrix, config, parameters.signalVariance)
            val centered = DoubleArray(data.responses.size) { data.responses[it] - parameters.mean }
            val alpha = factor.solve(centered)
            val quadratic = koblas.vectorKernels.dot(centered, 0, alpha, 0, alpha.size)
            val likelihood = if (alpha.isEmpty()) {
                0.0
            } else {
                -0.5 *
                    (
                        quadratic + factor.logDeterminant() + alpha.size * ln(
                            2.0 * PI,
                        )
                        )
            }
            require(likelihood.isFinite() && alpha.all { it.isFinite() }) { "Nonfinite GP fit; rescale responses" }
            return GaussianProcessModel(data, config, parameters, factor, alpha, jitter, likelihood)
        }
    }
}

private fun covariance(
    left: DenseMatrix,
    right: DenseMatrix,
    kernel: GaussianKernel,
    parameters: GaussianProcessParameters,
): DenseMatrix {
    val result = DenseMatrix.zero(left.rows, right.rows)
    for (i in 0 until left.rows) {
        for (j in 0 until right.rows) {
            var square = 0.0
            for (column in 0 until left.cols) {
                val distance = (left[i, column] - right[j, column]) / parameters.lengthScales[column]
                square += distance * distance
            }
            result[i, j] = when {
                !square.isFinite() -> 0.0

                kernel == GaussianKernel.Rbf -> parameters.signalVariance * exp(-0.5 * square)

                else -> {
                    val radius = sqrt(5.0) * sqrt(square)
                    if (radius > 745.0) {
                        0.0
                    } else {
                        parameters.signalVariance * (1.0 + radius + radius * radius / 3.0) * exp(
                            -radius,
                        )
                    }
                }
            }
        }
    }
    return result
}

private fun stabilized(
    matrix: DenseMatrix,
    config: GaussianProcessConfig,
    signalVariance: Double,
): Pair<CholeskyFactor, Double> {
    val scale = maxOf(signalVariance, (0 until matrix.rows).maxOfOrNull { matrix[it, it] } ?: signalVariance)
    var jitter = scale * config.jitter
    var failure: NotPositiveDefinite? = null
    repeat(config.factorizationAttempts) {
        val regularized = DenseMatrix.wrap(matrix.rows, matrix.cols, matrix.values.copyOf())
        for (i in 0 until matrix.rows) regularized[i, i] += jitter
        try {
            return regularized.factorCholesky() to jitter
        } catch (error: NotPositiveDefinite) {
            failure = error
            jitter *= 10.0
        }
    }
    throw requireNotNull(failure)
}
