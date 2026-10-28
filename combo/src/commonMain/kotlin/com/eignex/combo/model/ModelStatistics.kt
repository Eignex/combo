package com.eignex.combo.model

import com.eignex.combo.util.currentTimeNanos
import com.eignex.klause.model.FloatSpec
import com.eignex.klause.model.IntSpec
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/** Named native accumulators whose snapshots supply the model's statistical inputs. */
class ModelStatistics(private val definition: ModelDefinition) : AutoCloseable {
    private val statistics = definition.statistics.mapValues { StatisticResource.create(it.value) }

    /** Feed one named accumulator using the same native fields as a standalone statistic. */
    fun observe(name: String, observation: JsonObject): JsonElement =
        requireNotNull(statistics[name]) { "unknown statistic '$name'" }.update(observation)

    /** Read every accumulator once, at the same timestamp, without advancing it. */
    fun snapshot(timestampNanos: Long = currentTimeNanos()): JsonObject =
        JsonObject(statistics.mapValues { it.value.snapshot(timestampNanos) })

    /** Reset all named accumulators to their native baselines. */
    fun reset(): JsonObject = JsonObject(statistics.mapValues { it.value.reset() })

    /** Freeze one snapshot per statistic, then lower ordinary Klause constraints against it. */
    fun resolve(timestampNanos: Long = currentTimeNanos()): ResolvedStatisticalModel {
        require(!definition.learned) { "learned models resolve statistics as context" }
        requireEvidence()
        return definition.resolveStatistics(snapshot(timestampNanos))
    }

    /** Freeze statistical context for one learned decision; retain it with any resulting proposal. */
    fun context(timestampNanos: Long = currentTimeNanos()): JsonObject {
        require(definition.learned || definition.problem != null) {
            "statistical context requires a learned or native model"
        }
        requireEvidence()
        val snapshots = snapshot(timestampNanos)
        return JsonObject(
            definition.inputs.mapValues { (name, input) ->
                val primitive = input.readPrimitive(snapshots, requireNotNull(input.value))
                if (definition.problem != null) return@mapValues primitive
                val value = primitive.content.toDouble()
                when (val spec = definition.space!!.context.getValue(name)) {
                    is IntSpec -> {
                        require(value.toInt().toDouble() == value && value >= spec.min && value <= spec.max) {
                            "'$name' must resolve to an exact integer within its context bounds"
                        }
                        JsonPrimitive(value.toInt())
                    }

                    is FloatSpec -> {
                        require(value in spec.min..spec.max) { "'$name' outside its context bounds" }
                        JsonPrimitive(value)
                    }

                    else -> error("'$name' must bind a numeric context")
                }
            },
        )
    }

    private fun requireEvidence() {
        for (input in definition.inputs.values) {
            if (statistics.getValue(input.statistic).observationCount == 0L) {
                throw InsufficientStatisticalEvidence(
                    "observe '${input.statistic}' before resolving statistical inputs",
                )
            }
        }
    }

    override fun close() = statistics.values.forEach { it.close() }
}

/** A statistical input has not received a positive-weight observation yet. */
class InsufficientStatisticalEvidence(message: String) : IllegalStateException(message)
