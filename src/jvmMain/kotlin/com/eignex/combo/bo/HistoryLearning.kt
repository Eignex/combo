package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.FeatureEncoder
import com.eignex.combo.shared.LearningFeedback
import com.eignex.koblas.DenseVector
import com.eignex.koblas.Vector
import com.eignex.kumulant.core.SeriesStat
import com.eignex.kumulant.stat.regression.tree.RegressionObservation
import kotlin.random.Random

internal class HistoryLearning<Model : Any>(
    private val projection: FeatureEncoder<Vector>,
    private val store: ObservationStore,
    private val fit: (List<RegressionObservation>, Random) -> Model,
    private val mean: (Model, Vector) -> Double,
    private val randomSeed: Int,
    rewards: SeriesStat<*>?,
    trainAbsError: SeriesStat<*>?,
    testAbsError: SeriesStat<*>?,
) {
    private var cached: Model? = null
    private var fittedRevision: Long? = null
    var fitCount = 0L
        private set

    private val feedback = LearningFeedback(
        encode = { sample, context ->
            require(projection.space.matches(sample.sample, projection.space.assumptionsFor(context))) {
                "sample contextual values do not match the supplied context"
            }
            val features = projection.encode(sample, context).toDoubleArray()
            require(features.all { it.isFinite() }) { "features must be finite" }
            Input(sample, context, DenseVector.wrap(features))
        },
        mean = { input: Input -> mean(model(), input.features) },
        record = { input, reward, weight -> store.append(Observation(input.sample, input.context, reward, weight)) },
        rewards = rewards,
        trainAbsError = trainAbsError,
        testAbsError = testAbsError,
    )

    fun model(): Model = if (fittedRevision == store.revision) requireNotNull(cached) else model(store.snapshot())

    fun model(history: ObservationHistory): Model {
        if (fittedRevision != history.revision) {
            cached = fit(history, Random(randomSeed.toLong() xor history.revision), bootstrap = false)
            fittedRevision = history.revision
        }
        return requireNotNull(cached)
    }

    fun fit(history: ObservationHistory, random: Random, bootstrap: Boolean): Model {
        val rows = history.observations.map { observation ->
            require(
                projection.space.matches(
                    observation.sample.sample,
                    projection.space.assumptionsFor(observation.context),
                ),
            ) {
                "stored sample does not match its context"
            }
            RegressionObservation(
                DenseVector.wrap(projection.encode(observation.sample, observation.context).toDoubleArray()),
                observation.reward,
                observation.weight,
            )
        }
        val training = if (bootstrap && rows.isNotEmpty()) List(rows.size) { rows[random.nextInt(rows.size)] } else rows
        fitCount++
        return fit(training, random)
    }

    fun predict(sample: BanditSample, context: Context): Double = feedback.predict(sample, context)

    fun train(sample: BanditSample, context: Context, reward: Double, weight: Double) {
        if (validWeight(reward, weight)) feedback.train(sample, context, reward, weight)
    }

    fun update(sample: BanditSample, context: Context, reward: Double, weight: Double) {
        if (validWeight(reward, weight)) feedback.update(sample, context, reward, weight)
    }

    private fun validWeight(reward: Double, weight: Double): Boolean {
        require(weight >= 0.0 && weight.isFinite()) { "history weights must be nonnegative and finite" }
        require(reward.isFinite()) { "reward must be finite" }
        return weight > 0.0
    }

    private data class Input(val sample: BanditSample, val context: Context, val features: Vector)
}
