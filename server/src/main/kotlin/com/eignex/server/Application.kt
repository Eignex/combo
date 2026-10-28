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
import io.ktor.server.response.respond
import io.ktor.server.routing.get
import io.ktor.server.routing.routing
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.serialization.SerializationException

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
        modelRoutes(state)
    }
}

internal fun ApplicationCall.resourceId(): String = parameters["id"].orEmpty()
