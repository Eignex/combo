package com.eignex.server

import com.eignex.combo.model.ModelDefinition
import com.eignex.combo.model.ModelQuery
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.server.request.receive
import io.ktor.server.response.respond
import io.ktor.server.routing.*
import kotlinx.serialization.json.*

internal fun Route.modelRoutes(state: ServerState) {
    route("/v1/models") {
        get { call.respond(state.models.list().map { mapOf("id" to it.id) }) }
        post {
            val definition = call.receive<ModelDefinition>()
            val info = state.models.create { ModelResource(definition) }
            call.response.headers.append(HttpHeaders.Location, "/v1/models/${info.id}")
            call.respond(HttpStatusCode.Created, mapOf("id" to info.id))
        }
        get("/{id}") {
            val timestamp = call.request.queryParameters["timestampNanos"]?.let {
                requireNotNull(it.toLongOrNull()) { "timestampNanos must be an integer" }
            } ?: System.nanoTime()
            call.respond(state.models.access(call.resourceId()) { model, _ -> model.describe(timestamp) })
        }
        post("/{id}/recommend") {
            val query = call.receive<ModelQuery>()
            call.respond(state.models.access(call.resourceId(), query.timeoutMillis) { model, control -> model.recommend(query, control) })
        }
        post("/{id}/evaluate") {
            val query = call.receive<ModelQuery>()
            call.respond(state.models.access(call.resourceId(), query.timeoutMillis) { model, control -> model.evaluate(query, control) })
        }
        post("/{id}/propose") {
            val request = call.receive<JsonObject>()
            val timeout = request["timeoutMillis"]?.jsonPrimitive?.longOrNull ?: 30_000
            call.respond(state.models.access(call.resourceId(), timeout) { model, control -> model.propose(request, control) })
        }
        post("/{id}/observe") {
            val request = call.receive<JsonObject>()
            call.respond(state.models.access(call.resourceId()) { model, _ -> model.observe(request) })
        }
        post("/{id}/reset") {
            call.respond(state.models.access(call.resourceId()) { model, _ -> model.reset() })
        }
        delete("/{id}/proposals/{proposal}") {
            val proposal = requireNotNull(call.parameters["proposal"]?.toLongOrNull()) { "invalid proposal identifier" }
            call.respond(state.models.access(call.resourceId()) { model, _ -> model.cancel(proposal) })
        }
        delete("/{id}") {
            state.models.delete(call.resourceId())
            call.respond(HttpStatusCode.NoContent)
        }
    }
}
