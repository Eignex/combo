package com.eignex.server

import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.application.Application
import io.ktor.server.application.ApplicationCall
import io.ktor.server.application.ApplicationStopped
import io.ktor.server.application.install
import io.ktor.server.application.log
import io.ktor.server.cio.CIO
import io.ktor.server.engine.embeddedServer
import io.ktor.server.plugins.BadRequestException
import io.ktor.server.plugins.PayloadTooLargeException
import io.ktor.server.plugins.bodylimit.RequestBodyLimit
import io.ktor.server.plugins.contentnegotiation.ContentNegotiation
import io.ktor.server.plugins.statuspages.StatusPages
import io.ktor.server.request.receive
import io.ktor.server.response.respond
import io.ktor.server.routing.delete
import io.ktor.server.routing.get
import io.ktor.server.routing.post
import io.ktor.server.routing.route
import io.ktor.server.routing.routing
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.encodeToJsonElement
import kotlinx.serialization.json.put

/** Start the API-only server for reusable solver and learner state. */
fun main() {
    val host = System.getenv("EIGNEX_SERVER_HOST") ?: "127.0.0.1"
    val port = System.getenv("EIGNEX_SERVER_PORT")?.toInt() ?: 8080
    embeddedServer(CIO, host = host, port = port) { resourceApi() }.start(wait = true)
}

internal fun Application.resourceApi(state: ServerState = ServerState()) {
    monitor.subscribe(ApplicationStopped) { state.close() }
    install(ContentNegotiation) { json(wireJson) }
    install(RequestBodyLimit) { bodyLimit { 2 * 1024 * 1024L } }
    install(StatusPages) {
        exception<PayloadTooLargeException> { call, _ ->
            call.respond(HttpStatusCode.PayloadTooLarge, ApiError("body_too_large", "request body exceeds 2 MiB"))
        }
        exception<BadRequestException> { call, _ ->
            call.respond(HttpStatusCode.BadRequest, ApiError("invalid_request", "invalid request body"))
        }
        exception<SerializationException> { call, _ ->
            call.respond(HttpStatusCode.BadRequest, ApiError("invalid_request", "invalid request body"))
        }
        exception<IllegalArgumentException> { call, error ->
            call.respond(HttpStatusCode.BadRequest, ApiError("invalid_request", error.message ?: "invalid request"))
        }
        exception<ApiException> { call, error ->
            if (error.status == HttpStatusCode.ServiceUnavailable) call.response.headers.append(HttpHeaders.RetryAfter, "1")
            call.respond(error.status, ApiError(error.code, error.message.orEmpty()))
        }
        exception<TimeoutCancellationException> { call, _ ->
            call.respond(HttpStatusCode.RequestTimeout, ApiError("operation_timed_out", "operation deadline exceeded"))
        }
        exception<Throwable> { call, error ->
            if (error is CancellationException) throw error
            this@resourceApi.log.error("Resource operation failed", error)
            call.respond(HttpStatusCode.InternalServerError, ApiError("execution_failed", "operation failed"))
        }
    }
    routing {
        get("/health") { call.respond(mapOf("status" to "ok")) }
        route("/v1/problems") {
            get { call.respond(state.problems.list()) }
            post {
                val request = call.receive<CreateProblem>()
                call.created("problems", state.problems.create(request.timeoutMillis) { ProblemResource(request) })
            }
            get("/{id}") {
                val id = call.resourceId()
                call.respond(state.problems.access(id) { resource, _ -> resource.describe(id) })
            }
            delete("/{id}") {
                state.problems.delete(call.resourceId())
                call.respond(HttpStatusCode.NoContent)
            }
            post("/{id}/solve") {
                val request = call.receive<ProblemQuery>()
                call.respond(state.problems.access(call.resourceId(), request.timeoutMillis) { resource, control ->
                    resource.solve(request, control)
                })
            }
            post("/{id}/sample") {
                val request = call.receive<ProblemQuery>()
                call.respond(state.problems.access(call.resourceId(), request.timeoutMillis) { resource, control ->
                    resource.sample(request, control)
                })
            }
        }
        route("/v1/statistics") {
            get { call.respond(state.statistics.list()) }
            post {
                call.receive<CreateStatistics>()
                call.created("statistics", state.statistics.create { StatisticsResource() })
            }
            get("/{id}") {
                val id = call.resourceId()
                call.respond(state.statistics.access(id) { resource, _ -> buildJsonObject {
                    put("id", id)
                    put("type", resource.type)
                    put("snapshot", wireJson.encodeToJsonElement(resource.snapshot()))
                } })
            }
            delete("/{id}") {
                state.statistics.delete(call.resourceId())
                call.respond(HttpStatusCode.NoContent)
            }
            get("/{id}/snapshot") {
                call.respond(state.statistics.access(call.resourceId()) { resource, _ -> resource.snapshot() })
            }
            post("/{id}/updates") {
                val request = call.receive<StatisticsUpdate>()
                call.respond(state.statistics.access(call.resourceId()) { resource, control -> resource.update(request, control) })
            }
            post("/{id}/merge") {
                val request = call.receive<SummarySnapshot>()
                call.respond(state.statistics.access(call.resourceId()) { resource, control -> resource.merge(request, control) })
            }
            post("/{id}/reset") {
                call.respond(state.statistics.access(call.resourceId()) { resource, _ -> resource.reset() })
            }
        }
        route("/v1/optimizers") {
            get { call.respond(state.optimizers.list()) }
            post {
                val request = call.receive<CreateOptimizer>()
                call.created("optimizers", state.optimizers.create(request.timeoutMillis) { OptimizerResource(request) })
            }
            get("/{id}") {
                val id = call.resourceId()
                call.respond(state.optimizers.access(id) { resource, _ -> resource.describe(id) })
            }
            delete("/{id}") {
                state.optimizers.delete(call.resourceId())
                call.respond(HttpStatusCode.NoContent)
            }
            post("/{id}/suggestions") {
                val request = call.receive<SuggestionRequest>()
                call.respond(state.optimizers.access(call.resourceId(), request.timeoutMillis) { resource, control ->
                    resource.suggest(request, control)
                })
            }
            get("/{id}/trials") {
                call.respond(state.optimizers.access(call.resourceId()) { resource, _ -> resource.trials() })
            }
            post("/{id}/trials/{trialId}/complete") {
                val trialId = call.trialId()
                val request = call.receive<CompleteTrial>()
                call.respond(state.optimizers.access(call.resourceId()) { resource, _ -> resource.complete(trialId, request) })
            }
            post("/{id}/trials/{trialId}/fail") {
                val trialId = call.trialId()
                val request = call.receive<TrialReason>()
                call.respond(state.optimizers.access(call.resourceId()) { resource, _ -> resource.fail(trialId, request.reason) })
            }
            post("/{id}/trials/{trialId}/cancel") {
                val trialId = call.trialId()
                call.respond(state.optimizers.access(call.resourceId()) { resource, _ -> resource.cancel(trialId) })
            }
        }
    }
}

private fun ApplicationCall.resourceId(): String = parameters["id"].orEmpty()
private fun ApplicationCall.trialId(): Long = requireNotNull(parameters["trialId"]?.toLongOrNull()) { "invalid trial id" }
private suspend fun ApplicationCall.created(collection: String, resource: ResourceInfo) {
    response.headers.append(HttpHeaders.Location, "/v1/$collection/${resource.id}")
    respond(HttpStatusCode.Created, resource)
}
