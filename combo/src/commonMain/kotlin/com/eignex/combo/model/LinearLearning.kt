package com.eignex.combo.model

import com.eignex.combo.bandit.glm.LinearBandit
import com.eignex.combo.decisions.Context
import com.eignex.combo.shared.LinearFeatureProjection
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.solver.Sample
import com.eignex.klause.solver.objective.LinearObjective
import com.eignex.kumulant.core.SeriesStat
import com.eignex.kumulant.schema.runtime.materialize
import com.eignex.kumulant.schema.spec.BayesianRegression
import com.eignex.kumulant.schema.spec.RegressionStatSpec
import com.eignex.kumulant.stat.regression.glm.LinearPosterior
import com.eignex.kumulant.stat.regression.glm.MultivariateGaussian
import kotlinx.serialization.Serializable

/** Native Bayesian reward model and its independently specified coefficient sampling policy. */
@Serializable
data class LinearLearning(
    /** Native Kumulant regression specification, including its coefficient prior. */
    val regression: RegressionStatSpec<*>,
    /** Native Kumulant posterior sampler. */
    val posterior: LinearPosterior<*>,
    /** Nonnegative finite multiplier for posterior exploration. */
    val exploration: Double = 1.0,
) {
    init {
        require(regression is BayesianRegression) { "linear learning currently supports bayesianRegression" }
        require(posterior == MultivariateGaussian) { "Bayesian regression requires multivariateGaussian" }
        require(exploration.isFinite() && exploration >= 0.0) { "exploration must be nonnegative and finite" }
    }

    /** Materialize the native learner without translating its prior, features, or posterior. */
    fun materialize(
        projection: LinearFeatureProjection,
        innerOptimizer: (LinearObjective, Assumptions) -> Sample?,
        seed: Int,
        maximize: Boolean,
        rewards: SeriesStat<*>? = null,
        contextOptimizer: ((LinearObjective, Assumptions, Context) -> Sample?)? = null,
    ): LinearBandit<*> = LinearBandit(
        projection, (regression as BayesianRegression).materialize(), MultivariateGaussian, exploration,
        innerOptimizer, seed, maximize, rewards, contextOptimizer = contextOptimizer,
    )
}
