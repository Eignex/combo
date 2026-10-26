package com.eignex.server

import com.eignex.koblas.DenseVector
import com.eignex.koblas.Vector
import com.eignex.kumulant.bandit.ContextualBandit
import com.eignex.kumulant.bandit.PerArmBandit
import com.eignex.kumulant.bandit.TrackedContextualBandit
import com.eignex.kumulant.bandit.TrackedUnivariateBandit
import com.eignex.kumulant.bandit.UnivariateBandit
import com.eignex.kumulant.bandit.Bandit
import com.eignex.kumulant.bandit.contextual.ContextualBanditSpec
import com.eignex.kumulant.bandit.contextual.RegressionContextualBandit
import com.eignex.kumulant.bandit.materialize
import com.eignex.kumulant.bandit.univariate.UnivariateBanditSpec
import com.eignex.kumulant.core.*
import com.eignex.kumulant.schema.runtime.materialize
import com.eignex.kumulant.schema.spec.StatSpec
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.*
import kotlin.random.Random

internal sealed class StatsResource : ManagedResource {
    open val canChoose = false
    open fun series(): SeriesStat<*>? = null
    abstract fun snapshot(timestampNanos: Long = System.nanoTime()): JsonElement
    abstract fun update(observation: JsonObject): JsonElement
    abstract fun reset(): JsonElement

    open fun choose(request: JsonObject): JsonElement = throw IllegalArgumentException("choose requires a bandit")

    companion object {
        fun create(definition: JsonObject, seed: Int): StatsResource {
            val serializers = listOf(StatSpec.serializer(), UnivariateBanditSpec.serializer(), ContextualBanditSpec.serializer())
            for (serializer in serializers) {
                val spec = try {
                    wireJson.decodeFromJsonElement(serializer, definition)
                } catch (_: SerializationException) {
                    continue
                }
                val resource = when (spec) {
                    is StatSpec -> AccumulatorResource(spec.materialize())
                    is UnivariateBanditSpec -> BanditResource(spec.materialize(Random(seed), nowNanos = System::nanoTime))
                    is ContextualBanditSpec -> BanditResource(spec.materialize(Random(seed), nowNanos = System::nanoTime))
                    else -> error("unsupported Kumulant specification")
                }
                resource.snapshot()
                return resource
            }
            throw SerializationException("invalid Kumulant specification")
        }
    }
}

private class AccumulatorResource(private val stat: Stat<*>) : StatsResource() {
    override val type = "kumulant.stat"
    override fun series(): SeriesStat<*>? = stat as? SeriesStat<*>

    override fun snapshot(timestampNanos: Long): JsonElement = encodeResult(stat.read(timestampNanos))

    override fun update(observation: JsonObject): JsonElement {
        val weight = observation.number("weight", 1.0)
        val timestamp = observation.integer("timestampNanos", System.nanoTime())
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
        return snapshot(timestamp)
    }

    override fun reset(): JsonElement {
        stat.reset()
        return snapshot()
    }
}

private class BanditResource(private val bandit: Bandit) : StatsResource() {
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
        is PerArmBandit<*> -> put("arms", JsonArray(bandit.snapshot().map(::encodeResult)))
        is TrackedUnivariateBandit<*> -> {
            put("inner", bandit.inner.snapshotJson())
            bandit.chooseResult()?.let { put("choose", encodeResult(it)) }
            bandit.updateArmRewardResult()?.let { put("updateArmReward", encodeResult(it)) }
        }
        is TrackedContextualBandit<*> -> {
            put("inner", bandit.inner.snapshotJson())
            bandit.chooseResult()?.let { put("choose", encodeResult(it)) }
            bandit.updateJointResult()?.let { put("updateJoint", encodeResult(it)) }
            bandit.updateMarginalResult()?.let { put("updateMarginal", encodeResult(it)) }
            bandit.updateArmRewardResult()?.let { put("updateArmReward", encodeResult(it)) }
        }
        else -> error("unsupported bandit snapshot")
    }
    if (this@snapshotJson is RegressionContextualBandit<*>) {
        globalSnapshot()?.let { put("global", encodeResult(it)) }
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
