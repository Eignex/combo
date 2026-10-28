package com.eignex.combo.model

import com.eignex.combo.util.currentTimeNanos
import com.eignex.koblas.DenseVector
import com.eignex.koblas.Vector
import com.eignex.kumulant.bandit.Bandit
import com.eignex.kumulant.bandit.ContextualBandit
import com.eignex.kumulant.bandit.PerArmBandit
import com.eignex.kumulant.bandit.TrackedContextualBandit
import com.eignex.kumulant.bandit.TrackedUnivariateBandit
import com.eignex.kumulant.bandit.UnivariateBandit
import com.eignex.kumulant.bandit.contextual.ContextualBanditSpec
import com.eignex.kumulant.bandit.contextual.RegressionContextualBandit
import com.eignex.kumulant.bandit.materialize
import com.eignex.kumulant.bandit.univariate.UnivariateBanditSpec
import com.eignex.kumulant.core.*
import com.eignex.kumulant.schema.runtime.materialize
import com.eignex.kumulant.schema.spec.StatSpec
import com.eignex.skema.SchemaJson
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.*
import kotlin.random.Random

/** Native Kumulant accumulator or bandit with its existing JSON inputs and outputs. */
sealed class StatisticResource : AutoCloseable {
    /** Internal statistical modality, independent of the model's public capabilities. */
    abstract val type: String

    /** Successful positive-weight observations, used to reject uninitialized bound inputs. */
    open val observationCount: Long get() = 0

    /** Whether this resource selects bandit arms. */
    open val canChoose = false

    /** Scalar accumulator view for attaching an outcome tracker. */
    open fun series(): SeriesStat<*>? = null

    /** Read one native snapshot at the supplied monotonic timestamp. */
    abstract fun snapshot(timestampNanos: Long = currentTimeNanos()): JsonElement

    /** Feed an observation using the native modality fields. */
    abstract fun update(observation: JsonObject): JsonElement

    /** Reset to the native statistic baseline. */
    abstract fun reset(): JsonElement

    /** Choose an arm using an ordinary or contextual native bandit. */
    open fun choose(request: JsonObject): JsonElement = throw IllegalArgumentException("choose requires a bandit")

    override fun close() = Unit

    /** Construct resources with the native Kumulant serializers and materializers. */
    companion object {
        /** Construct a typed native accumulator without a JSON decoding round trip. */
        fun create(spec: StatSpec): StatisticResource = AccumulatorResource(spec.materialize())

        /** Decode the canonical accumulator or bandit vocabulary. */
        fun create(definition: JsonObject, seed: Int): StatisticResource {
            val serializers = listOf(
                StatSpec.serializer(),
                UnivariateBanditSpec.serializer(),
                ContextualBanditSpec.serializer(),
            )
            for (serializer in serializers) {
                val spec = try {
                    SchemaJson.decodeFromJsonElement(serializer, definition)
                } catch (_: SerializationException) {
                    continue
                }
                val resource = when (spec) {
                    is StatSpec -> AccumulatorResource(spec.materialize())

                    is UnivariateBanditSpec ->
                        BanditResource(spec.materialize(Random(seed), nowNanos = ::currentTimeNanos))

                    is ContextualBanditSpec ->
                        BanditResource(spec.materialize(Random(seed), nowNanos = ::currentTimeNanos))

                    else -> error("unsupported Kumulant specification")
                }
                resource.snapshot()
                return resource
            }
            throw SerializationException("invalid Kumulant specification")
        }
    }
}

private class AccumulatorResource(private val stat: Stat<*>) : StatisticResource() {
    override var observationCount: Long = 0
        private set
    override val type = "kumulant.stat"
    override fun series(): SeriesStat<*>? = stat as? SeriesStat<*>

    override fun snapshot(timestampNanos: Long): JsonElement = encodeStatisticResult(stat.read(timestampNanos))

    override fun update(observation: JsonObject): JsonElement {
        val weight = observation.number("weight", 1.0)
        val timestamp = observation.integer("timestampNanos", currentTimeNanos())
        when (stat) {
            is SeriesStat<*> -> {
                observation.only("value", "weight", "timestampNanos")
                stat.update(observation.number("value"), timestamp, weight)
            }

            is DiscreteStat<*> -> {
                observation.only("value", "weight", "timestampNanos")
                stat.update(observation.integer("value"), timestamp, weight)
            }

            is PairedStat<*> -> {
                observation.only("x", "y", "weight", "timestampNanos")
                stat.update(observation.number("x"), observation.number("y"), timestamp, weight)
            }

            is VectorStat<*> -> {
                observation.only("x", "weight", "timestampNanos")
                stat.update(observation.vector("x"), timestamp, weight)
            }

            is RegressionStat<*> -> {
                observation.only("x", "y", "weight", "timestampNanos")
                val x = observation.vector("x")
                require(x.size == stat.featureSize) { "x must contain ${stat.featureSize} features" }
                stat.update(x, observation.number("y"), timestamp, weight)
            }

            else -> error("unsupported accumulator modality")
        }
        if (weight > 0.0) observationCount++
        return snapshot(timestamp)
    }

    override fun reset(): JsonElement {
        stat.reset()
        observationCount = 0
        return snapshot()
    }
}

private class BanditResource(private val bandit: Bandit) : StatisticResource() {
    override val type = "kumulant.bandit"
    override val canChoose = true

    override fun snapshot(timestampNanos: Long): JsonElement = bandit.snapshotJson()

    override fun update(observation: JsonObject): JsonElement {
        val weight = observation.number("weight", 1.0)
        val arm = observation.integer("arm").also {
            require(it in 0L until bandit.nbrArms.toLong()) { "arm must be in [0, ${bandit.nbrArms})" }
        }.toInt()
        val reward = observation.number("reward")
        when (bandit) {
            is UnivariateBandit -> {
                observation.only("arm", "reward", "weight")
                bandit.update(arm, reward, weight)
            }

            is ContextualBandit -> {
                observation.only("arm", "x", "reward", "weight")
                bandit.update(arm, bandit.context(observation), reward, weight)
            }
        }
        return snapshot()
    }

    override fun choose(request: JsonObject): JsonElement {
        val arm = when (bandit) {
            is UnivariateBandit -> {
                request.only()
                bandit.choose()
            }

            is ContextualBandit -> {
                request.only("x")
                bandit.choose(bandit.context(request))
            }

            else -> error("unsupported bandit")
        }
        return buildJsonObject { put("arm", arm) }
    }

    override fun reset(): JsonElement {
        bandit.reset()
        return snapshot()
    }
}

private fun ContextualBandit.context(request: JsonObject): Vector {
    val size = when (this) {
        is RegressionContextualBandit<*> -> armStat(0).featureSize
        is TrackedContextualBandit<*> -> contextFeatureSize
        else -> error("unsupported contextual bandit")
    }
    return request.vector("x").also { require(it.size == size) { "x must contain $size features" } }
}

private fun Bandit.snapshotJson(): JsonObject = buildJsonObject {
    when (val bandit = this@snapshotJson) {
        is PerArmBandit<*> -> put("arms", JsonArray(bandit.snapshot().map(::encodeStatisticResult)))

        is TrackedUnivariateBandit<*> -> {
            put("inner", bandit.inner.snapshotJson())
            bandit.chooseResult()?.let { put("choose", encodeStatisticResult(it)) }
            bandit.updateArmRewardResult()?.let { put("updateArmReward", encodeStatisticResult(it)) }
        }

        is TrackedContextualBandit<*> -> {
            put("inner", bandit.inner.snapshotJson())
            bandit.chooseResult()?.let { put("choose", encodeStatisticResult(it)) }
            bandit.updateJointResult()?.let { put("updateJoint", encodeStatisticResult(it)) }
            bandit.updateMarginalResult()?.let { put("updateMarginal", encodeStatisticResult(it)) }
            bandit.updateArmRewardResult()?.let { put("updateArmReward", encodeStatisticResult(it)) }
        }

        else -> error("unsupported bandit snapshot")
    }
    if (this@snapshotJson is RegressionContextualBandit<*>) {
        globalSnapshot()?.let { put("global", encodeStatisticResult(it)) }
    }
}

private fun JsonObject.only(vararg fields: String) {
    val unexpected = keys - fields.toSet()
    require(unexpected.isEmpty()) { "unexpected fields: ${unexpected.joinToString()}" }
}

private fun JsonObject.number(name: String, default: Double? = null): Double =
    if (name !in this && default != null) default else get(name).finiteNumber(name)

private fun JsonElement?.finiteNumber(name: String): Double {
    val primitive = this as? JsonPrimitive
    return requireNotNull(primitive?.takeUnless { it.isString }?.doubleOrNull?.takeIf { it.isFinite() }) {
        "$name must be a finite number"
    }
}

private fun JsonObject.integer(name: String, default: Long? = null): Long {
    if (name !in this && default != null) return default
    val primitive = get(name) as? JsonPrimitive
    return requireNotNull(primitive?.takeUnless { it.isString }?.longOrNull) { "$name must be an integer" }
}

private fun JsonObject.vector(name: String): Vector {
    val values = get(name) as? JsonArray ?: throw IllegalArgumentException("$name must be an array of numbers")
    return DenseVector.wrap(DoubleArray(values.size) { values[it].finiteNumber("$name[$it]") })
}
