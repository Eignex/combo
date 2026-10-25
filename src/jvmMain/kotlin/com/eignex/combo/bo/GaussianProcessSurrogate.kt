package com.eignex.combo.bo

import com.eignex.combo.bo.model.GaussianProcessConfig
import com.eignex.combo.bo.model.GaussianProcessModel
import com.eignex.combo.bo.model.GaussianProcessParameters
import com.eignex.koblas.DenseMatrix
import kotlin.random.Random

/**
 * Exact full-history GP for low-dimensional numeric coordinates. Fits copy inputs and use bounded
 * marginal-likelihood search for covariance parameters. Use SearchTransform and response standardization
 * for numeric BO. Unit statistical weights are required; noise variances and replicate rows are explicit.
 * Factorization costs O(n³) time and O(n²) space. Hyperparameter uncertainty is not integrated out.
 *
 * @property config Mean, kernel, observation-noise, and numerical stabilization configuration.
 */
class GaussianProcessSurrogate(val config: GaussianProcessConfig = GaussianProcessConfig()) : Surrogate {
    override fun fit(data: TrainingData, random: Random): FittedGaussianProcess =
        FittedGaussianProcess(GaussianProcessModel.fit(data, config))
}

/**
 * GP posterior conditional on fitted parameters, with latent marginal and joint moments.
 * Conditioning freezes covariance parameters and the mean, and re-factorizes the augmented data.
 * Function draws sample joint latent values lazily and retain them across repeated/reordered queries.
 * Sampling adds numerical stabilization only; first visits to new coordinates determine seeded random
 * assignment order. Each draw is single-threaded and retains sampled points, so its storage and solve cost
 * grow with distinct queries.
 */
class FittedGaussianProcess internal constructor(private val model: GaussianProcessModel) :
    ConditionableSurrogate,
    JointPosteriorSurrogate {
    override val uncertaintyKind: UncertaintyKind = UncertaintyKind.LatentPosterior

    /** Independent fitted parameter snapshot in training-response coordinates. */
    val parameters: GaussianProcessParameters get() = model.parameters.copy(
        lengthScales = model.parameters.lengthScales.toList(),
    )

    /** Log marginal likelihood at the fitted covariance parameters, including numerical stabilization. */
    val logMarginalLikelihood: Double get() = model.logMarginalLikelihood

    /** Actual diagonal numerical stabilization used in fitting, separate from observation noise. */
    val numericalJitter: Double get() = model.numericalJitter

    override fun predict(features: DenseMatrix): PredictiveMoments = model.predict(features)
    override fun predictJoint(features: DenseMatrix): JointPredictiveMoments = model.predictJoint(features)
    override fun jointPosterior(baseline: DenseMatrix): JointPosterior = model.jointPosterior(baseline)
    override fun condition(temporary: TrainingData): FittedGaussianProcess = FittedGaussianProcess(
        model.condition(temporary),
    )
    override fun drawFunction(random: Random): SampledFunction = model.drawFunction(random)
}
