package com.eignex.server

import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.server.request.receive
import io.ktor.server.response.respond
import io.ktor.server.routing.*
import kotlinx.serialization.json.JsonObject

internal fun Route.statsRoutes(state: ServerState) {
    route("/v1/stats") {
        get { call.respond(state.stats.list().map { mapOf("id" to it.id) }) }
        post {
            val definition = call.receive<JsonObject>()
            val seed = call.request.queryParameters["seed"]?.let {
                requireNotNull(it.toIntOrNull()) { "seed must be an integer" }
            } ?: 0
            val info = state.stats.create { StatsResource.create(definition, seed) }
            call.response.headers.append(HttpHeaders.Location, "/v1/stats/${info.id}")
            call.respond(HttpStatusCode.Created, mapOf("id" to info.id))
        }
        get("/{id}") {
            val timestamp = call.request.queryParameters["timestampNanos"]?.let {
                requireNotNull(it.toLongOrNull()) { "timestampNanos must be an integer" }
            } ?: System.nanoTime()
            call.respond(state.stats.access(call.resourceId()) { resource, _ -> resource.snapshot(timestamp) })
        }
        post("/{id}/update") {
            val observation = call.receive<JsonObject>()
            call.respond(state.stats.access(call.resourceId()) { resource, _ -> resource.update(observation) })
        }
        post("/{id}/choose") {
            val request = call.receive<JsonObject>()
            call.respond(state.stats.access(call.resourceId()) { resource, _ -> resource.choose(request) })
        }
        post("/{id}/reset") {
            call.respond(state.stats.access(call.resourceId()) { resource, _ -> resource.reset() })
        }
        delete("/{id}") {
            state.stats.delete(call.resourceId())
            call.respond(HttpStatusCode.NoContent)
        }
    }
}
