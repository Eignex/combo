package com.eignex.combo.model

import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject

/** Named native accumulators whose snapshots supply the model's statistical inputs. */
class ModelStatistics(private val definition: ModelDefinition) : AutoCloseable {
    private val statistics = definition.statistics.mapValues { StatisticResource.create(it.value) }

    /** Feed one named accumulator using the same native fields as a standalone statistic. */
    fun observe(name: String, observation: JsonObject): JsonElement =
        requireNotNull(statistics[name]) { "unknown statistic '$name'" }.update(observation)

    /** Read every accumulator once, at the same timestamp, without advancing it. */
    fun snapshot(timestampNanos: Long = System.nanoTime()): JsonObject =
        JsonObject(statistics.mapValues { it.value.snapshot(timestampNanos) })

    /** Reset all named accumulators to their native baselines. */
    fun reset(): JsonObject = JsonObject(statistics.mapValues { it.value.reset() })

    /** Freeze one snapshot per statistic, then lower ordinary Klause constraints against it. */
    fun resolve(timestampNanos: Long = System.nanoTime()): ResolvedStatisticalModel {
        for (input in definition.inputs.values) {
            if (statistics.getValue(input.statistic).observationCount == 0L) {
                throw InsufficientStatisticalEvidence("observe '${input.statistic}' before resolving statistical inputs")
            }
        }
        return definition.resolveStatistics(snapshot(timestampNanos))
    }

    override fun close() = statistics.values.forEach { it.close() }
}

/** A statistical input has not received a positive-weight observation yet. */
class InsufficientStatisticalEvidence(message: String) : IllegalStateException(message)
