package com.eignex.combo.bo

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.Vector
import com.eignex.kumulant.stat.regression.tree.ForestPrediction
import com.eignex.kumulant.stat.regression.tree.expectedImprovement
import com.eignex.kumulant.stat.regression.tree.expectedImprovementLogNormal
import kotlin.random.Random

/** A fixed acquisition objective; its values are always maximized. */
fun interface Acquisition {
    /** Score feature rows together using fixed model and random draws. */
    fun evaluate(features: DenseMatrix): Vector
}

/** Inputs captured once when constructing an acquisition objective. */
class AcquisitionContext(
    /** Immutable fitted surrogate. */
    val model: FittedSurrogate,
    /** Improvement baseline in original reward units; null when none has been evaluated in context. */
    val incumbent: Double?,
    /** Objective and noise policy. */
    val outcome: OutcomeSpec,
)

/** Bind a model and baseline into an independently optimizable acquisition function. */
fun interface AcquisitionFactory {
    /** Randomness must be fixed here, rather than redrawn for each candidate evaluation. */
    fun create(context: AcquisitionContext, random: Random): Acquisition
}

/** Analytic Gaussian EI approximation; a missing contextual baseline uses posterior-mean search. */
class ExpectedImprovement(
    /** Improvement margin in original reward units. */
    val margin: Double = 0.0,
) : AcquisitionFactory {
    init {
        require(margin >= 0.0 && margin.isFinite())
    }
    override fun create(context: AcquisitionContext, random: Random): Acquisition = Acquisition { features ->
        val logarithmic = context.model is LogResponseSurrogate
        val moments = if (context.model is LogResponseSurrogate) {
            context.model.predictLog(features)
        } else {
            context.model.predict(features)
        }
        DenseVector.of(
            DoubleArray(features.rows) { i ->
                context.incumbent?.let {
                    val prediction = ForestPrediction(moments.means[i], moments.variances[i])
                    if (logarithmic) {
                        prediction.expectedImprovementLogNormal(it, context.outcome.maximize, margin)
                    } else {
                        prediction.expectedImprovement(it, context.outcome.maximize, margin)
                    }
                } ?: context.outcome.score(
                    if (logarithmic) {
                        kotlin.math.exp(
                            moments.means[i] + moments.variances[i] / 2.0,
                        )
                    } else {
                        moments.means[i]
                    },
                )
            },
        )
    }
}

/** Mean reward acquisition, also used to optimize a fixed bootstrap model. */
object PosteriorMean : AcquisitionFactory {
    override fun create(context: AcquisitionContext, random: Random): Acquisition = Acquisition { features ->
        val means = context.model.predict(features).means
        DenseVector.of(DoubleArray(means.size) { context.outcome.score(means[it]) })
    }
}

/** Thompson acquisition from one coherent function draw over all candidate evaluations. */
object PosteriorThompson : AcquisitionFactory {
    override fun create(context: AcquisitionContext, random: Random): Acquisition {
        val draw = context.model.drawFunction(random)
        return Acquisition { features ->
            val values = draw.evaluate(features)
            DenseVector.of(DoubleArray(values.size) { context.outcome.score(values[it]) })
        }
    }
}
