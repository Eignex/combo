package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.FeatureEncoder
import com.eignex.combo.shared.LearningFeedback
import com.eignex.combo.shared.assignmentKey
import com.eignex.koblas.DenseVector
import com.eignex.koblas.Vector
import com.eignex.kumulant.core.PairedStat
import com.eignex.kumulant.core.SeriesStat
import kotlin.random.Random

internal class SurrogateLearning(
    private val projection: FeatureEncoder<Vector>,
    private val store: ObservationStore,
    private val surrogate: Surrogate,
    private val outcome: OutcomeSpec,
    private val seed: Int,
    rewards: SeriesStat<*>?,
    trainScore: PairedStat<*>?,
    testScore: PairedStat<*>?,
) {
    private var cached: FittedSurrogate? = null
    private var fittedRevision: Long? = null
    var fitCount = 0L
        private set

    private data class Input(val sample: BanditSample, val context: Context, val features: Vector, val noise: Double?)
    private fun input(sample: BanditSample, context: Context, noise: Double? = null): Input {
        require(
            projection.space.matches(sample.sample, projection.space.assumptionsFor(context)),
        ) { "context mismatch" }
        val features = DenseVector.of(projection.encode(sample, context).toDoubleArray())
        require(features.size == projection.featureSize && features.toDoubleArray().all { it.isFinite() })
        return Input(sample, context, features, noise)
    }
    private val feedback = LearningFeedback(
        encode = { sample, context -> input(sample, context) },
        mean = { value: Input -> model(store.snapshot()).mean(value.features) },
        record = { value, reward, weight ->
            store.append(Observation(value.sample, value.context, reward, weight, value.noise))
        },
        rewards = rewards,
        trainScore = trainScore,
        testScore = testScore,
    )

    private fun trainingSeed(observations: List<Observation>): Long {
        var key = seed.toLong()
        for (row in observations) {
            key = 31L * key + row.sample.sample.assignmentKey().hashCode()
            key = 31L * key + row.reward.toBits()
            key = 31L * key + row.weight.toBits()
            key = 31L * key + (row.noiseVariance?.toBits() ?: 0L)
        }
        return key
    }

    fun model(history: ObservationHistory): FittedSurrogate {
        if (fittedRevision != history.revision) {
            cached = fit(history.observations, Random(trainingSeed(history.observations)))
            fittedRevision = history.revision
        }
        return requireNotNull(cached)
    }

    fun fit(
        observations: List<Observation>,
        random: Random,
        bootstrap: Boolean = false,
        response: ResponseTransform = ResponseTransform.from(observations, outcome),
    ): FittedSurrogate {
        val rows = if (bootstrap && observations.isNotEmpty()) {
            List(observations.size) {
                observations[random.nextInt(observations.size)]
            }
        } else {
            observations
        }
        val transform = response
        val data = TrainingData(
            featureRows(rows.map { input(it.sample, it.context).features }, projection.featureSize),
            DenseVector.of(DoubleArray(rows.size) { transform.encode(rows[it].reward) }),
            DenseVector.of(DoubleArray(rows.size) { rows[it].weight }),
            DenseVector.of(
                DoubleArray(
                    rows.size,
                ) {
                    rows[it].noiseVariance?.let { noise ->
                        transform.encodeNoise(
                            rows[it].reward,
                            noise,
                        )
                    } ?: Double.NaN
                },
            ),
        )
        fitCount++
        return transform.model(surrogate.fit(data, random))
    }

    fun temporary(history: ObservationHistory, observations: List<Observation>, random: Random): FittedSurrogate {
        val base = model(history)
        if (base is ConditionableSurrogate) {
            return base.condition(
                TrainingData(
                    featureRows(observations.map { input(it.sample, it.context).features }, projection.featureSize),
                    DenseVector.of(DoubleArray(observations.size) { observations[it].reward }),
                    DenseVector.of(DoubleArray(observations.size) { observations[it].weight }),
                    DenseVector.of(DoubleArray(observations.size) { observations[it].noiseVariance ?: Double.NaN }),
                ),
            )
        }
        return fit(
            history.observations + observations,
            random,
            response = ResponseTransform.from(history.observations, outcome),
        )
    }

    fun predict(sample: BanditSample, context: Context): Double = feedback.predict(sample, context)
    fun train(sample: BanditSample, context: Context, reward: Double, weight: Double, noise: Double? = null) {
        if (valid(reward, weight)) {
            val value = input(sample, context, noise)
            store.append(Observation(value.sample, context, reward, weight, noise))
        }
    }
    fun update(
        sample: BanditSample,
        context: Context,
        reward: Double,
        weight: Double,
        noise: Double? = null,
        onRecorded: () -> Unit = {},
    ) {
        if (valid(reward, weight)) {
            Observation(sample, context, reward, weight, noise)
            feedback.updateEncoded(input(sample, context, noise), reward, weight, onRecorded)
        }
    }
    private fun valid(reward: Double, weight: Double): Boolean {
        require(reward.isFinite() && weight >= 0.0 && weight.isFinite())
        require(outcome.responseScale != ResponseScale.Log || reward > 0.0) { "log responses must be positive" }
        return weight > 0.0
    }
}

internal fun FittedSurrogate.mean(features: Vector): Double = predict(
    featureRows(listOf(features), features.size),
).means[0]
