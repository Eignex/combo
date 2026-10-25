package com.eignex.server

import com.eignex.combo.decisions.DecisionSpaceDef
import com.eignex.klause.model.SchemaEntry
import com.eignex.skema.SchemaDef
import com.eignex.skema.schemaJsonConfig
import io.ktor.http.HttpStatusCode
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject

internal val wireJson = Json { schemaJsonConfig() }
internal val emptyObject = JsonObject(emptyMap())

internal fun validateTimeout(value: Long) {
    require(value in 1..300_000) { "timeoutMillis must be between 1 and 300000" }
}

@Serializable
internal data class CreateProblem(val schema: SchemaDef<SchemaEntry>, val timeoutMillis: Long = 30_000) {
    init { validateTimeout(timeoutMillis) }
}

@Serializable
internal data class ProblemQuery(
    val assumptions: JsonObject = emptyObject,
    val maxDecisions: Long = 100_000,
    val seed: Long = 0,
    val timeoutMillis: Long = 30_000,
) {
    init {
        validateTimeout(timeoutMillis)
        require(maxDecisions in 1..10_000_000) { "maxDecisions must be between 1 and 10000000" }
    }
}

@Serializable
internal data class CreateOptimizer(
    val space: DecisionSpaceDef,
    val seed: Int = 0,
    val maximize: Boolean = true,
    val timeoutMillis: Long = 30_000,
) {
    init { validateTimeout(timeoutMillis) }
}

@Serializable
internal data class SuggestionRequest(
    val context: JsonObject = emptyObject,
    val count: Int = 1,
    val timeoutMillis: Long = 30_000,
) {
    init {
        validateTimeout(timeoutMillis)
        require(count in 1..64) { "count must be between 1 and 64" }
    }
}

@Serializable
internal data class CompleteTrial(
    val reward: Double,
    val weight: Double = 1.0,
    val noiseVariance: Double? = null,
) {
    init {
        require(reward.isFinite()) { "reward must be finite" }
        require(weight.isFinite() && weight > 0.0) { "weight must be positive and finite" }
        require(noiseVariance == null || (noiseVariance.isFinite() && noiseVariance >= 0.0)) { "invalid noise variance" }
    }
}

@Serializable
internal data class TrialReason(val reason: String) {
    init { require(reason.isNotBlank()) { "reason must not be blank" } }
}

@Serializable
internal data class ResourceInfo(val id: String, val type: String)

@Serializable
internal data class ApiError(val code: String, val message: String)

internal class ApiException(val status: HttpStatusCode, val code: String, message: String) : RuntimeException(message)
internal fun notFound(): Nothing = throw ApiException(HttpStatusCode.NotFound, "not_found", "unknown resource")
internal fun capacityExceeded(): Nothing = throw ApiException(
    HttpStatusCode.ServiceUnavailable, "capacity_exceeded", "server capacity reached",
)
