package com.eignex.server

import com.eignex.combo.decisions.DecisionSpaceDef
import com.eignex.klause.model.BoolRef
import com.eignex.klause.model.BoolSpec
import com.eignex.klause.model.FloatSpec
import com.eignex.klause.model.Iff
import com.eignex.klause.model.IntSpec
import com.eignex.klause.model.NamedConstraint
import com.eignex.klause.model.Not
import com.eignex.skema.SchemaDef
import io.ktor.client.HttpClient
import io.ktor.client.request.delete
import io.ktor.client.request.get
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.contentType
import io.ktor.server.testing.testApplication
import kotlinx.coroutines.delay
import kotlinx.coroutines.withTimeout
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class ApplicationTest {
    @Test
    fun `summary jobs can be submitted polled listed and deleted`() = testApplication {
        application { jobApi() }
        val response = client.post("/v1/jobs") {
            contentType(ContentType.Application.Json)
            setBody("""{"request":{"type":"kumulant.summary","values":[1,2,3,4]}}""")
        }
        assertEquals(HttpStatusCode.Accepted, response.status)
        val created = wireJson.decodeFromString<JobSnapshot>(response.bodyAsText())
        assertEquals("/v1/jobs/${created.id}", response.headers[HttpHeaders.Location])
        val completed = client.awaitJob(created.id)
        assertEquals(JobStatus.Succeeded, completed.status)
        assertNotNull(completed.startedAt)
        assertNotNull(completed.finishedAt)
        val result = assertNotNull(completed.result)
        assertEquals(2.5, result.getValue("mean").jsonPrimitive.double)
        assertEquals(1.25, result.getValue("variance").jsonPrimitive.double)
        val listed = wireJson.decodeFromString<List<JobSnapshot>>(client.get("/v1/jobs").bodyAsText())
        assertEquals(listOf(completed), listed)
        assertEquals(HttpStatusCode.NoContent, client.delete("/v1/jobs/${created.id}").status)
        assertEquals(HttpStatusCode.NotFound, client.get("/v1/jobs/${created.id}").status)
    }

    @Test
    fun `Klause returns a satisfying named assignment`() = testApplication {
        application { jobApi() }
        val request = KlauseSolve(SchemaDef(mapOf(
            "flag" to BoolSpec,
            "required" to NamedConstraint(BoolRef("flag")),
            "size" to IntSpec(2, 4),
        )))
        val completed = client.submitAndAwait(request)
        assertEquals(JobStatus.Succeeded, completed.status)
        val result = assertNotNull(completed.result)
        assertEquals("sat", result.getValue("verdict").jsonPrimitive.content)
        assertTrue(result.getValue("values").jsonObject.getValue("flag").jsonPrimitive.boolean)
    }

    @Test
    fun `Klause infeasibility is a successful computation with an unsat verdict`() = testApplication {
        application { jobApi() }
        val completed = client.submitAndAwait(KlauseSolve(SchemaDef(mapOf(
            "flag" to BoolSpec,
            "yes" to NamedConstraint(BoolRef("flag")),
            "no" to NamedConstraint(Not(BoolRef("flag"))),
        ))))
        assertEquals(JobStatus.Succeeded, completed.status)
        assertEquals("unsat", completed.result?.getValue("verdict")?.jsonPrimitive?.content)
    }

    @Test
    fun `unknown schema references report invalid job input`() = testApplication {
        application { jobApi() }
        val completed = client.submitAndAwait(KlauseSolve(SchemaDef(mapOf(
            "invalid" to NamedConstraint(BoolRef("missing")),
        ))))
        assertEquals(JobStatus.Failed, completed.status)
        assertEquals("invalid_job", completed.error?.code)
    }

    @Test
    fun `Combo suggestions preserve hard context and exclude duplicate reservations`() = testApplication {
        application { jobApi() }
        val request = ComboSuggest(
            DecisionSpaceDef(
                variables = mapOf("flag" to BoolSpec, "size" to IntSpec(0, 3)),
                context = mapOf("premium" to BoolSpec),
                constraints = mapOf("match" to Iff(BoolRef("flag"), BoolRef("premium"))),
            ),
            context = JsonObject(mapOf("premium" to JsonPrimitive(true))),
            count = 3,
        )
        val completed = client.submitAndAwait(request)
        assertEquals(JobStatus.Succeeded, completed.status, completed.error?.message)
        val suggestions = assertNotNull(completed.result).getValue("suggestions").jsonArray
        assertEquals(3, suggestions.size)
        assertEquals(3, suggestions.distinct().size)
        assertTrue(suggestions.all { it.jsonObject.getValue("flag").jsonPrimitive.boolean })
        assertTrue(suggestions.all { it.jsonObject.getValue("premium").jsonPrimitive.boolean })
    }

    @Test
    fun `Combo imports validated observations before suggesting`() = testApplication {
        application { jobApi() }
        val request = ComboSuggest(
            DecisionSpaceDef(variables = mapOf("size" to IntSpec(0, 8))),
            observations = (0..4).map {
                ObservationInput(JsonObject(mapOf("size" to JsonPrimitive(it))), reward = it.toDouble())
            },
        )
        val completed = client.submitAndAwait(request)
        assertEquals(JobStatus.Succeeded, completed.status, completed.error?.message)
        assertEquals(1, completed.result?.getValue("suggestions")?.jsonArray?.size)
    }

    @Test
    fun `float decision spaces produce bounded feasible suggestions`() = testApplication {
        application { jobApi() }
        val completed = client.submitAndAwait(ComboSuggest(
            DecisionSpaceDef(variables = mapOf("rate" to FloatSpec(0.1, 0.9))), count = 2,
        ))
        assertEquals(JobStatus.Succeeded, completed.status, completed.error?.message)
        val suggestions = assertNotNull(completed.result).getValue("suggestions").jsonArray
        assertEquals(2, suggestions.size)
        assertTrue(suggestions.all { it.jsonObject.getValue("rate").jsonPrimitive.double in 0.1..0.9 })
    }

    @Test
    fun `request bodies exceeding the configured limit are rejected`() = testApplication {
        application { jobApi() }
        val response = client.post("/v1/jobs") {
            contentType(ContentType.Application.Json)
            setBody(" ".repeat(2 * 1024 * 1024 + 1))
        }
        assertEquals(HttpStatusCode.PayloadTooLarge, response.status)
        assertEquals("body_too_large", wireJson.decodeFromString<ApiError>(response.bodyAsText()).code)
    }

    @Test
    fun `full queue returns retryable capacity error and cancellation releases its slot`() = testApplication {
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        val jobs = JobRegistry(workers = 1, queueCapacity = 1, execute = { _, _ ->
            started.countDown()
            release.await()
            JsonObject(emptyMap())
        })
        application { jobApi(jobs) }
        suspend fun submit() = client.post("/v1/jobs") {
            contentType(ContentType.Application.Json)
            setBody("""{"request":{"type":"kumulant.summary","values":[1]}}""")
        }
        try {
            assertEquals(HttpStatusCode.Accepted, submit().status)
            assertTrue(started.await(5, TimeUnit.SECONDS))
            val queued = wireJson.decodeFromString<JobSnapshot>(submit().bodyAsText())
            val rejected = submit()
            assertEquals(HttpStatusCode.ServiceUnavailable, rejected.status)
            assertEquals("1", rejected.headers[HttpHeaders.RetryAfter])
            assertEquals(HttpStatusCode.Conflict, client.delete("/v1/jobs/${queued.id}").status)
            val cancelled = client.post("/v1/jobs/${queued.id}/cancel")
            assertEquals(JobStatus.Cancelled, wireJson.decodeFromString<JobSnapshot>(cancelled.bodyAsText()).status)
            assertEquals(HttpStatusCode.Accepted, submit().status)
        } finally {
            release.countDown()
            jobs.close()
        }
    }

    @Test
    fun `malformed unsupported and invalid jobs are rejected before submission`() = testApplication {
        application { jobApi() }
        val bodies = listOf(
            "{",
            """{"request":{"type":"unknown"}}""",
            """{"request":{"type":"kumulant.summary","values":[]}}""",
            """{"request":{"type":"kumulant.summary","values":[1],"typo":true}}""",
            """{"request":{"type":"kumulant.summary","values":[1]},"timeoutMillis":0}""",
        )
        for (body in bodies) {
            val response = client.post("/v1/jobs") {
                contentType(ContentType.Application.Json)
                setBody(body)
            }
            assertEquals(HttpStatusCode.BadRequest, response.status, body)
            assertEquals("invalid_request", wireJson.decodeFromString<ApiError>(response.bodyAsText()).code)
        }
        assertEquals("[]", client.get("/v1/jobs").bodyAsText())
    }

    @Test
    fun `infeasible observation fails its job without breaking later jobs`() = testApplication {
        application { jobApi() }
        val failed = client.submitAndAwait(ComboSuggest(
            DecisionSpaceDef(
                variables = mapOf("flag" to BoolSpec), constraints = mapOf("yes" to BoolRef("flag")),
            ),
            observations = listOf(ObservationInput(JsonObject(mapOf("flag" to JsonPrimitive(false))), 1.0)),
        ))
        assertEquals(JobStatus.Failed, failed.status)
        assertEquals("invalid_job", failed.error?.code)
        assertEquals(JobStatus.Succeeded, client.submitAndAwait(KumulantSummary(listOf(1.0))).status)
    }

    @Test
    fun `missing jobs return not found on every job route`() = testApplication {
        application { jobApi() }
        assertEquals(HttpStatusCode.NotFound, client.get("/v1/jobs/missing").status)
        assertEquals(HttpStatusCode.NotFound, client.post("/v1/jobs/missing/cancel").status)
        assertEquals(HttpStatusCode.NotFound, client.delete("/v1/jobs/missing").status)
        assertEquals(HttpStatusCode.OK, client.get("/health").status)
    }
}

internal suspend fun HttpClient.submitAndAwait(request: JobRequest): JobSnapshot {
    val response = post("/v1/jobs") {
        contentType(ContentType.Application.Json)
        setBody(wireJson.encodeToString(Submission(request)))
    }
    assertEquals(HttpStatusCode.Accepted, response.status, response.bodyAsText())
    return awaitJob(wireJson.decodeFromString<JobSnapshot>(response.bodyAsText()).id)
}

internal suspend fun HttpClient.awaitJob(id: String): JobSnapshot = withTimeout(10_000) {
    var snapshot: JobSnapshot
    do {
        delay(5)
        snapshot = wireJson.decodeFromString<JobSnapshot>(get("/v1/jobs/$id").bodyAsText())
    } while (!snapshot.status.terminal)
    snapshot
}
