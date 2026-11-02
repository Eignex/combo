package com.eignex.combo.bench.diagnostics

import com.eignex.combo.bo.*
import com.eignex.combo.bo.model.*
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.koblas
import com.eignex.kumulant.math.factorCholesky
import com.eignex.kumulant.math.nextNormal
import com.eignex.kumulant.stat.summary.VarianceStat
import kotlin.math.*
import kotlin.random.Random

internal fun matrix(rows: List<DoubleArray>, dimensions: Int): DenseMatrix =
    if (rows.isEmpty()) DenseMatrix.zero(0, dimensions) else DenseMatrix.ofRows(rows.toTypedArray())

internal fun training(x: DenseMatrix, y: DoubleArray, noise: DoubleArray) = TrainingData(
    x, DenseVector.of(y), DenseVector.of(DoubleArray(y.size) { 1.0 }), DenseVector.of(noise),
)

/** Research-only Gaussian component: a proper N(0, meanVariance) constant mean is integrated exactly. */
internal class IntegratedMeanGp(
    input: TrainingData,
    scales: List<Double>,
    val signal: Double,
    val noise: Double,
    val meanVariance: Double = 1.0,
) : ConditionableSurrogate, JointPosteriorSurrogate {
    val scales = scales.toList()
    private val data = training(DenseMatrix.wrap(input.features.rows, input.features.cols, input.features.values.copyOf()),
        input.responses.toDoubleArray(), input.noiseVariances?.toDoubleArray() ?: DoubleArray(input.responses.size) { Double.NaN })
    override val uncertaintyKind = UncertaintyKind.LatentPosterior
    private fun kernel(a: DenseMatrix, b: DenseMatrix): DenseMatrix {
        val result = DenseMatrix.zero(a.rows, b.rows)
        for (i in 0 until a.rows) for (j in 0 until b.rows) {
            var square = 0.0
            for (d in scales.indices) square += ((a[i, d] - b[j, d]) / scales[d]).pow(2)
            val r = sqrt(5.0 * square)
            result[i, j] = signal * (1.0 + r + r * r / 3.0) * exp(-r) + meanVariance
        }
        return result
    }
    val jitter: Double
    private val factor: com.eignex.kumulant.math.CholeskyFactor
    private val alpha: DoubleArray
    val logLikelihood: Double
    init {
        require(input.features.cols == scales.size && scales.all { it > 0.0 && it.isFinite() })
        require(signal > 0.0 && signal.isFinite() && noise >= 0.0 && noise.isFinite() && meanVariance >= 0.0 && meanVariance.isFinite())
        require(input.weights.toDoubleArray().all { it == 1.0 })
        val covariance = kernel(data.features, data.features)
        for (i in 0 until covariance.rows) covariance[i, i] += data.noiseVariances!![i].takeUnless { it.isNaN() } ?: noise
        jitter = maxOf(signal + meanVariance, (0 until covariance.rows).maxOfOrNull { covariance[it, it] } ?: 1.0) * 1e-10
        for (i in 0 until covariance.rows) covariance[i, i] += jitter
        factor = covariance.factorCholesky()
        alpha = factor.solve(data.responses.toDoubleArray())
        logLikelihood = -0.5 * (data.responses.toDoubleArray().zip(alpha).sumOf { it.first * it.second } +
            factor.logDeterminant() + data.responses.size * ln(2.0 * PI))
        check(logLikelihood.isFinite())
    }
    override fun predictJoint(features: DenseMatrix): JointPredictiveMoments = jointFeatures(features).first
    private fun jointFeatures(features: DenseMatrix): Pair<JointPredictiveMoments, DenseMatrix> {
        require(features.cols == scales.size && features.values.all { it.isFinite() })
        val cross = kernel(data.features, features)
        val mean = koblas.gemv(cross, alpha, transpose = true)
        val solved = factor.solveLower(cross)
        val covariance = kernel(features, features)
        koblas.gemm(-1.0, solved, true, solved, false, 1.0, covariance)
        for (i in 0 until covariance.rows) covariance[i, i] = maxOf(0.0, covariance[i, i])
        return JointPredictiveMoments(DenseVector.of(mean), covariance) to solved
    }
    override fun jointPosterior(baseline: DenseMatrix): JointPosterior {
        val fixed = DenseMatrix.wrap(baseline.rows, baseline.cols, baseline.values.copyOf())
        val (moments, baselineSolved) = jointFeatures(fixed)
        return object : JointPosteriorNumerics {
            override val baseline = moments
            override val covarianceScale = signal + meanVariance
            override fun predict(features: DenseMatrix): JointPosteriorPrediction {
                val (candidate, solved) = jointFeatures(features)
                val cross = kernel(fixed, features)
                koblas.gemm(-1.0, baselineSolved, true, solved, false, 1.0, cross)
                return JointPosteriorPrediction(candidate, cross)
            }
        }
    }
    override fun predict(features: DenseMatrix): PredictiveMoments {
        require(features.cols == scales.size && features.values.all { it.isFinite() })
        // Marginals avoid an O(probes^2) joint covariance in calibration captures.
        val cross = kernel(data.features, features)
        val mean = koblas.gemv(cross, alpha, transpose = true)
        val solved = factor.solveLower(cross)
        return PredictiveMoments(DenseVector.of(mean), DenseVector.of(DoubleArray(features.rows) { j ->
            maxOf(0.0, signal + meanVariance - (0 until solved.rows).sumOf { solved[it, j] * solved[it, j] })
        }))
    }
    override fun condition(temporary: TrainingData): IntegratedMeanGp {
        require(temporary.weights.toDoubleArray().all { it == 1.0 })
        require(temporary.features.cols == scales.size)
        val rows = (0 until data.features.rows).map { i -> DoubleArray(scales.size) { data.features[i, it] } } +
            (0 until temporary.features.rows).map { i -> DoubleArray(scales.size) { temporary.features[i, it] } }
        return IntegratedMeanGp(training(matrix(rows, scales.size),
            data.responses.toDoubleArray() + temporary.responses.toDoubleArray(),
            data.noiseVariances!!.toDoubleArray() + (temporary.noiseVariances?.toDoubleArray() ?: DoubleArray(temporary.responses.size) { Double.NaN })),
            scales, signal, noise, meanVariance)
    }
    override fun drawFunction(random: Random): SampledFunction {
        val stream = Random(random.nextLong())
        var current = this
        val cache = linkedMapOf<List<Double>, Double>()
        return SampledFunction { features ->
            val keys = (0 until features.rows).map { i -> List(features.cols) { features[i, it].let { v -> if (v == 0.0) 0.0 else v } } }
            val missing = keys.distinct().filterNot(cache::containsKey)
            if (missing.isNotEmpty()) {
                val x = matrix(missing.map { it.toDoubleArray() }, features.cols)
                val moments = current.predictJoint(x)
                val covariance = DenseMatrix.wrap(x.rows, x.rows, moments.covariance.values.copyOf())
                for (i in 0 until x.rows) covariance[i, i] += 1e-10 * (signal + meanVariance)
                val draw = koblas.gemv(covariance.factorCholesky().lowerFactor(), DoubleArray(x.rows) { stream.nextNormal() })
                val y = DoubleArray(x.rows) { moments.means[it] + draw[it] }
                missing.forEachIndexed { i, key -> cache[key] = y[i] }
                current = current.condition(training(x, y, DoubleArray(y.size)))
            }
            DenseVector.of(DoubleArray(keys.size) { cache.getValue(keys[it]) })
        }
    }
}

internal class ParameterMixture(components: List<IntegratedMeanGp>, weights: DoubleArray) : ConditionableSurrogate {
    val components = components.toList()
    val weights = weights.copyOf()
    override val uncertaintyKind = UncertaintyKind.LatentPosterior
    init { require(components.size == weights.size && abs(weights.sum() - 1.0) < 1e-10) }
    override fun predict(features: DenseMatrix): PredictiveMoments {
        val predictions = components.map { it.predict(features) }
        val means = DoubleArray(features.rows) { i -> predictions.indices.sumOf { weights[it] * predictions[it].means[i] } }
        val variances = DoubleArray(features.rows) { i -> predictions.indices.sumOf {
            weights[it] * (predictions[it].variances[i] + (predictions[it].means[i] - means[i]).pow(2))
        } }
        return PredictiveMoments(DenseVector.of(means), DenseVector.of(variances))
    }
    override fun predictJoint(features: DenseMatrix): JointPredictiveMoments {
        val predictions = components.map { it.predictJoint(features) }
        val means = DoubleArray(features.rows) { i -> predictions.indices.sumOf { weights[it] * predictions[it].means[i] } }
        val covariance = DenseMatrix.zero(features.rows)
        for (i in means.indices) for (j in means.indices) covariance[i, j] = predictions.indices.sumOf {
            weights[it] * (predictions[it].covariance[i, j] + (predictions[it].means[i] - means[i]) * (predictions[it].means[j] - means[j]))
        }
        return JointPredictiveMoments(DenseVector.of(means), covariance)
    }
    override fun condition(temporary: TrainingData): ParameterMixture {
        val conditioned = components.map { it.condition(temporary) }
        return ParameterMixture(conditioned, normalizedLogWeights(DoubleArray(weights.size) {
            ln(weights[it]) + conditioned[it].logLikelihood - components[it].logLikelihood
        }))
    }
    override fun drawFunction(random: Random): SampledFunction = components[choose(weights, random.nextDouble())].drawFunction(random)
}

internal fun choose(weights: DoubleArray, uniform: Double): Int {
    var sum = 0.0
    for (i in weights.indices) { sum += weights[i]; if (uniform < sum) return i }
    return weights.lastIndex
}
internal fun normalizedLogWeights(logs: DoubleArray): DoubleArray {
    val maximum = logs.max()
    val values = DoubleArray(logs.size) { exp(logs[it] - maximum) }
    val total = values.sum()
    return DoubleArray(values.size) { values[it] / total }
}

/** Only benchmark code uses this prototype; existing fixed-parameter library contracts are untouched. */
internal fun parameterGrid(data: TrainingData, nodes: Int, selectMap: Boolean = false): ParameterMixture {
    require(nodes in listOf(3, 5))
    val unknown = data.noiseVariances!!.toDoubleArray().any { it.isNaN() }
    val axes = data.features.cols + 1 + if (unknown) 1 else 0
    val components = mutableListOf<IntegratedMeanGp>()
    val logs = mutableListOf<Double>()
    fun visit(values: List<Double>, prior: Double) {
        val axis = values.size
        if (axis == axes) {
            val fit = IntegratedMeanGp(data, values.take(data.features.cols), values[data.features.cols],
                if (unknown) values.last() else 0.0)
            components += fit
            logs += fit.logLikelihood + prior
            return
        }
        val center = if (axis < data.features.cols) ln(0.5) else if (axis == data.features.cols) 0.0 else ln(0.01)
        val radius = if (axis <= data.features.cols) ln(4.0) else ln(100.0)
        val deviation = if (axis < data.features.cols) 1.0 else 2.0
        for (i in 0 until nodes) {
            val offset = radius * (2.0 * i / (nodes - 1) - 1.0)
            visit(values + exp(center + offset), prior - 0.5 * (offset / deviation).pow(2) +
                if (i == 0 || i == nodes - 1) ln(0.5) else 0.0)
        }
    }
    visit(emptyList(), 0.0)
    val weights = normalizedLogWeights(logs.toDoubleArray())
    if (selectMap) {
        val best = weights.indices.maxBy { weights[it] }
        return ParameterMixture(listOf(components[best]), doubleArrayOf(1.0))
    }
    return ParameterMixture(components, weights)
}

internal class AffinePosterior(val base: FittedSurrogate, val center: Double, val scale: Double) : ConditionableSurrogate, JointPosteriorSurrogate {
    override val uncertaintyKind get() = base.uncertaintyKind
    override fun predict(features: DenseMatrix): PredictiveMoments = base.predict(features).let { p ->
        PredictiveMoments(DenseVector.of(DoubleArray(p.means.size) { center + scale * p.means[it] }),
            DenseVector.of(DoubleArray(p.means.size) { scale * scale * p.variances[it] }))
    }
    override fun predictJoint(features: DenseMatrix): JointPredictiveMoments? = base.predictJoint(features)?.let { p ->
        JointPredictiveMoments(DenseVector.of(DoubleArray(p.means.size) { center + scale * p.means[it] }),
            DenseMatrix.wrap(p.means.size, p.means.size, DoubleArray(p.covariance.values.size) { scale * scale * p.covariance.values[it] }))
    }
    override fun jointPosterior(baseline: DenseMatrix): JointPosterior? {
        val posterior = base.jointPosterior(baseline) ?: return null
        fun decoded(p: JointPredictiveMoments) = JointPredictiveMoments(
            DenseVector.of(DoubleArray(p.means.size) { center + scale * p.means[it] }),
            DenseMatrix.wrap(p.means.size, p.means.size, DoubleArray(p.covariance.values.size) { scale * scale * p.covariance.values[it] }))
        return object : JointPosteriorNumerics {
            override val baseline = decoded(posterior.baseline)
            override val covarianceScale = (posterior as? JointPosteriorNumerics)?.covarianceScale?.let { it * scale * scale } ?: 0.0
            override fun predict(features: DenseMatrix): JointPosteriorPrediction {
                val p = posterior.predict(features)
                return JointPosteriorPrediction(decoded(p.candidates), DenseMatrix.wrap(p.crossCovariance.rows, p.crossCovariance.cols,
                    DoubleArray(p.crossCovariance.values.size) { scale * scale * p.crossCovariance.values[it] }))
            }
        }
    }
    override fun condition(temporary: TrainingData): AffinePosterior {
        require(temporary.weights.toDoubleArray().all { it == 1.0 })
        return AffinePosterior(
        (base as ConditionableSurrogate).condition(training(temporary.features,
            DoubleArray(temporary.responses.size) { (temporary.responses[it] - center) / scale },
            DoubleArray(temporary.responses.size) { (temporary.noiseVariances?.get(it) ?: Double.NaN) / (scale * scale) })), center, scale)
    }
    override fun drawFunction(random: Random): SampledFunction {
        val draw = base.drawFunction(random)
        return SampledFunction { features -> draw.evaluate(features).let { y -> DenseVector.of(DoubleArray(y.size) { center + scale * y[it] }) } }
    }
}

internal class DiagnosticSurrogate(val policy: String, val trials: Int, val learnNoise: Boolean, val gridNodes: Int, val fitSeed: Int = 0) : Surrogate {
    lateinit var last: AffinePosterior
    var candidates = 0L
    var fitNanos = 0L
    var selectedNoise = 0.0
    var likelihood = 0.0
    var fitScore = 0.0
    var numericalJitter = 0.0
    var parameters = ""
    override fun fit(data: TrainingData, random: Random): FittedSurrogate {
        val stream = if (fitSeed == 0) random else Random(random.nextLong() xor fitSeed.toLong())
        val start = System.nanoTime()
        val moments = VarianceStat()
        data.responses.toDoubleArray().forEach { moments.update(it) }
        val estimate = moments.read()
        val center = estimate.mean
        val scale = sqrt(estimate.variance).takeIf { it.isFinite() && it > 0.0 } ?: 1.0
        val encoded = training(data.features, DoubleArray(data.responses.size) { (data.responses[it] - center) / scale },
            DoubleArray(data.responses.size) { (data.noiseVariances?.get(it) ?: Double.NaN) / (scale * scale) })
        val model: FittedSurrogate
        if (policy in listOf("mixture", "grid-map")) {
            val mixture = parameterGrid(encoded, gridNodes, policy == "grid-map")
            model = mixture
            val unknown = encoded.noiseVariances!!.toDoubleArray().any { it.isNaN() }
            candidates += gridNodes.toDouble().pow(data.features.cols + 1 + if (unknown) 1 else 0).toLong()
            selectedNoise = mixture.components.indices.sumOf { mixture.weights[it] * mixture.components[it].noise } * scale * scale
            likelihood = mixture.components.indices.sumOf { mixture.weights[it] * mixture.components[it].logLikelihood }
            fitScore = likelihood
            numericalJitter = mixture.components.maxOf { it.jitter }
            parameters = mixture.components.indices.joinToString(";") { "${mixture.weights[it]}:${mixture.components[it].scales}:${mixture.components[it].signal}:${mixture.components[it].noise}" }
        } else {
            val fitting = GaussianProcessFitting(
                meanEstimation = if (policy in listOf("mean", "combined")) GaussianMeanEstimation.CovarianceWeighted else GaussianMeanEstimation.Empirical,
                regularization = if (policy in listOf("prior", "combined")) GaussianProcessRegularization() else null,
                search = if (policy in listOf("multistart", "combined")) GaussianHyperparameterSearch.Multistart else GaussianHyperparameterSearch.Coordinate)
            val gp = GaussianProcessSurrogate(fitting, GaussianProcessConfig(hyperparameterTrials = trials, learnNoise = learnNoise)).fit(encoded, stream)
            model = gp
            candidates += trials + 1
            selectedNoise = if (data.noiseVariances!!.toDoubleArray().none { it.isNaN() })
                data.noiseVariances!!.toDoubleArray().average() else gp.parameters.noiseVariance * scale * scale
            likelihood = gp.logMarginalLikelihood
            fitScore = gp.logFitScore
            numericalJitter = gp.numericalJitter
            parameters = gp.parameters.toString()
        }
        last = AffinePosterior(model, center, scale)
        fitNanos += System.nanoTime() - start
        return last
    }
    fun components(): List<Pair<FittedSurrogate, Double>> {
        val model = last.base
        return if (model is ParameterMixture) model.components.indices.map {
            AffinePosterior(model.components[it], last.center, last.scale) to model.weights[it]
        } else listOf(last to 1.0)
    }
    fun jointFactory(noisy: Boolean, draws: Int): JointAcquisitionFactory = JointAcquisitionFactory { context, random ->
        val seed = random.nextLong()
        val components = components()
        val objectives = components.map { (model, _) ->
            val input = JointAcquisitionContext(model, context.incumbent, context.outcome, context.baseline, context.pending)
            // Integrate component EI itself. Moment-matching the mixture is not mixture EI.
            if (noisy) JointLogNoisyExpectedImprovement(draws).create(input, Random(seed))
            else JointLogExpectedImprovement(draws).create(input, Random(seed))
        }
        JointAcquisition { features ->
            val values = DoubleArray(objectives.size) { ln(components[it].second) + objectives[it].evaluate(features) }
            val maximum = values.max()
            if (maximum == Double.NEGATIVE_INFINITY) maximum else maximum + ln(values.sumOf { exp(it - maximum) })
        }
    }
}
