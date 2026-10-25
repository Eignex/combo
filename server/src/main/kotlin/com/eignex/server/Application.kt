package com.eignex.server

import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.serialization.kotlinx.json.json
import io.ktor.server.application.Application
import io.ktor.server.application.ApplicationStopped
import io.ktor.server.application.install
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
import kotlinx.serialization.SerializationException

/** Start the API-only job server. */
fun main() {
    val host = System.getenv("EIGNEX_SERVER_HOST") ?: "127.0.0.1"
    val port = System.getenv("EIGNEX_SERVER_PORT")?.toInt() ?: 8080
    embeddedServer(CIO, host = host, port = port) { jobApi() }.start(wait = true)
}

internal fun Application.jobApi(jobs: JobRegistry = JobRegistry()) {
    monitor.subscribe(ApplicationStopped) { jobs.close() }
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
        exception<JobCapacityExceeded> { call, _ ->
            call.response.headers.append(HttpHeaders.RetryAfter, "1")
            call.respond(HttpStatusCode.ServiceUnavailable, ApiError("capacity_exceeded", "job capacity reached"))
        }
        exception<JobConflict> { call, error ->
            call.respond(HttpStatusCode.Conflict, ApiError("job_conflict", error.message ?: "job conflict"))
        }
    }
    routing {
        get("/health") { call.respond(mapOf("status" to "ok")) }
        route("/v1/jobs") {
            post {
                val job = jobs.submit(call.receive<Submission>())
                call.response.headers.append(HttpHeaders.Location, "/v1/jobs/${job.id}")
                call.respond(HttpStatusCode.Accepted, job)
            }
            get { call.respond(jobs.list()) }
            get("/{id}") {
                val job = jobs.get(call.parameters["id"].orEmpty())
                if (job == null) call.respond(HttpStatusCode.NotFound, ApiError("not_found", "unknown job"))
                else call.respond(job)
            }
            post("/{id}/cancel") {
                val job = jobs.cancel(call.parameters["id"].orEmpty())
                if (job == null) call.respond(HttpStatusCode.NotFound, ApiError("not_found", "unknown job"))
                else call.respond(job)
            }
            delete("/{id}") {
                if (jobs.delete(call.parameters["id"].orEmpty())) call.respond(HttpStatusCode.NoContent)
                else call.respond(HttpStatusCode.NotFound, ApiError("not_found", "unknown job"))
            }
        }
    }
}
