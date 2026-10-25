package com.eignex.server

import com.eignex.combo.decisions.DecisionSpaceDef
import com.eignex.klause.model.SchemaEntry
import com.eignex.skema.SchemaDef
import com.eignex.skema.schemaJsonConfig
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonClassDiscriminator
import kotlinx.serialization.json.JsonObject

internal val wireJson = Json { schemaJsonConfig() }

@Serializable
internal data class Submission(val request: JobRequest, val timeoutMillis: Long = 30_000) {
    init {
        require(timeoutMillis in 1..300_000) { "timeoutMillis must be between 1 and 300000" }
    }
}

@OptIn(ExperimentalSerializationApi::class)
@Serializable
@JsonClassDiscriminator("type")
internal sealed interface JobRequest {
    val operation: String
}

@Serializable
@SerialName("klause.solve")
internal data class KlauseSolve(
    val schema: SchemaDef<SchemaEntry>,
    val maxDecisions: Long = 100_000,
    val seed: Long = 0,
) : JobRequest {
    override val operation: String get() = "klause.solve"
    init { require(maxDecisions in 1..10_000_000) { "maxDecisions must be between 1 and 10000000" } }
}

@Serializable
@SerialName("kumulant.summary")
internal data class KumulantSummary(val values: List<Double>) : JobRequest {
    override val operation: String get() = "kumulant.summary"
    init {
        require(values.isNotEmpty() && values.size <= 100_000) { "values must contain 1 to 100000 observations" }
        require(values.all(Double::isFinite)) { "values must be finite" }
    }
}

@Serializable
@SerialName("combo.suggest")
internal data class ComboSuggest(
    val space: DecisionSpaceDef,
    val observations: List<ObservationInput> = emptyList(),
    val context: JsonObject = JsonObject(emptyMap()),
    val count: Int = 1,
    val seed: Int = 0,
    val maximize: Boolean = true,
) : JobRequest {
    override val operation: String get() = "combo.suggest"
    init {
        require(count in 1..64) { "count must be between 1 and 64" }
        require(observations.size <= 10_000) { "at most 10000 observations are supported" }
    }
}

@Serializable
internal data class ObservationInput(
    val values: JsonObject,
    val reward: Double,
    val context: JsonObject = JsonObject(emptyMap()),
    val weight: Double = 1.0,
) {
    init {
        require(reward.isFinite()) { "reward must be finite" }
        require(weight.isFinite() && weight > 0.0) { "weight must be positive and finite" }
    }
}

@Serializable
internal enum class JobStatus {
    @SerialName("queued") Queued,
    @SerialName("running") Running,
    @SerialName("succeeded") Succeeded,
    @SerialName("failed") Failed,
    @SerialName("cancelled") Cancelled,
    @SerialName("timed_out") TimedOut;

    val terminal: Boolean get() = this != Queued && this != Running
}

@Serializable
internal data class JobSnapshot(
    val id: String,
    val operation: String,
    val status: JobStatus,
    val submittedAt: String,
    val startedAt: String? = null,
    val finishedAt: String? = null,
    val result: JsonObject? = null,
    val error: ApiError? = null,
)

@Serializable
internal data class ApiError(val code: String, val message: String)
