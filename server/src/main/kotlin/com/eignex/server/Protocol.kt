package com.eignex.server

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
internal data class ResourceInfo(val id: String, val type: String)

@Serializable
internal data class ApiError(val code: String, val message: String)

internal class ApiException(val status: HttpStatusCode, val code: String, message: String) : RuntimeException(message)
internal fun notFound(): Nothing = throw ApiException(HttpStatusCode.NotFound, "not_found", "unknown resource")
internal fun capacityExceeded(): Nothing = throw ApiException(
    HttpStatusCode.ServiceUnavailable, "capacity_exceeded", "server capacity reached",
)
