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
import io.ktor.client.statement.HttpResponse
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.contentType
import io.ktor.server.testing.testApplication
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class ApplicationTest {
    @Test
    fun `one compiled problem serves repeated queries with independent assumptions`() = testApplication {
        application { resourceApi() }
        val request = CreateProblem(SchemaDef(mapOf("flag" to BoolSpec)))
        val problem = client.create("problems", wireJson.encodeToString(request))
        for (value in listOf(true, false, true)) {
            val response = client.send("/v1/problems/${problem.id}/solve", """{"assumptions":{"flag":$value}}""")
            assertEquals(HttpStatusCode.OK, response.status, response.bodyAsText())
            val result = response.objectBody()
            assertEquals("sat", result.getValue("verdict").jsonPrimitive.content)
            assertEquals(value, result.getValue("values").jsonObject.getValue("flag").jsonPrimitive.boolean)
        }
        val sample = client.send("/v1/problems/${problem.id}/sample", """{"assumptions":{"flag":true}}""")
        assertEquals(HttpStatusCode.OK, sample.status)
        assertTrue(sample.objectBody().getValue("values").jsonObject.getValue("flag").jsonPrimitive.boolean)
        assertEquals(request.schema, wireJson.decodeFromString<CreateProblem>(
            JsonObject(mapOf("schema" to client.get("/v1/problems/${problem.id}").objectBody().getValue("schema"))).toString(),
        ).schema)
        assertEquals(1, wireJson.decodeFromString<List<ResourceInfo>>(client.get("/v1/problems").bodyAsText()).size)
    }

    @Test
    fun `infeasibility is a solver verdict rather than a failed transport operation`() = testApplication {
        application { resourceApi() }
        val problem = client.create("problems", wireJson.encodeToString(CreateProblem(SchemaDef(mapOf(
            "flag" to BoolSpec,
            "yes" to NamedConstraint(BoolRef("flag")),
            "no" to NamedConstraint(Not(BoolRef("flag"))),
        )))))
        val result = client.send("/v1/problems/${problem.id}/solve", "{}")
        assertEquals(HttpStatusCode.OK, result.status)
        assertEquals("unsat", result.objectBody().getValue("verdict").jsonPrimitive.content)
    }

    @Test
    fun `optimizer reservations exclude pending configurations across requests`() = testApplication {
        application { resourceApi() }
        val optimizer = client.optimizer(DecisionSpaceDef(variables = mapOf("size" to IntSpec(0, 3))))
        val first = client.suggest(optimizer.id, 2)
        val second = client.suggest(optimizer.id, 2)
        val all = first + second
        assertEquals(4, all.map { it.jsonObject.getValue("id") }.distinct().size)
        assertEquals(4, all.map { it.jsonObject.getValue("values") }.distinct().size)
        assertEquals(0, client.suggest(optimizer.id, 1).size)
        val description = client.get("/v1/optimizers/${optimizer.id}").objectBody()
        assertEquals(4, description.getValue("pendingCount").jsonPrimitive.int)
        assertEquals(0, description.getValue("observationCount").jsonPrimitive.int)
        assertEquals(4, client.get("/v1/optimizers/${optimizer.id}/trials").arrayBody().size)
    }

    @Test
    fun `optimizer completions retain history and reject duplicate feedback`() = testApplication {
        application { resourceApi() }
        val optimizer = client.optimizer(DecisionSpaceDef(variables = mapOf("size" to IntSpec(0, 10))))
        repeat(5) {
            val trial = client.suggest(optimizer.id, 1).single().jsonObject
            val id = trial.getValue("id").jsonPrimitive.long
            val result = client.send("/v1/optimizers/${optimizer.id}/trials/$id/complete", """{"reward":$it}""")
            assertEquals(HttpStatusCode.OK, result.status, result.bodyAsText())
            assertEquals("completed", result.objectBody().getValue("status").jsonPrimitive.content)
            assertEquals(HttpStatusCode.Conflict,
                client.send("/v1/optimizers/${optimizer.id}/trials/$id/complete", """{"reward":9}""").status)
        }
        assertEquals(1, client.suggest(optimizer.id, 1).size)
        val description = client.get("/v1/optimizers/${optimizer.id}").objectBody()
        assertEquals(5, description.getValue("observationCount").jsonPrimitive.int)
        assertEquals(1, description.getValue("pendingCount").jsonPrimitive.int)
    }

    @Test
    fun `continuous trials accept feedback by identifier without reconstructing assignments`() = testApplication {
        application { resourceApi() }
        val optimizer = client.optimizer(DecisionSpaceDef(variables = mapOf("rate" to FloatSpec(0.1, 0.9))))
        val trial = client.suggest(optimizer.id, 1).single().jsonObject
        val id = trial.getValue("id").jsonPrimitive.long
        val completed = client.send("/v1/optimizers/${optimizer.id}/trials/$id/complete", """{"reward":0.7,"noiseVariance":0.01}""")
        assertEquals(HttpStatusCode.OK, completed.status, completed.bodyAsText())
        assertEquals(trial.getValue("values"), completed.objectBody().getValue("values"))
        assertEquals(1, client.suggest(optimizer.id, 1).size)
        assertEquals(1, client.get("/v1/optimizers/${optimizer.id}").objectBody().getValue("observationCount").jsonPrimitive.int)
    }

    @Test
    fun `each suggestion has its own hard context on the same optimizer`() = testApplication {
        application { resourceApi() }
        val optimizer = client.optimizer(DecisionSpaceDef(
            variables = mapOf("flag" to BoolSpec), context = mapOf("premium" to BoolSpec),
            constraints = mapOf("match" to Iff(BoolRef("flag"), BoolRef("premium"))),
        ))
        for (value in listOf(true, false)) {
            val response = client.send("/v1/optimizers/${optimizer.id}/suggestions", """{"context":{"premium":$value}}""")
            assertEquals(HttpStatusCode.OK, response.status, response.bodyAsText())
            val trial = response.objectBody().getValue("trials").jsonArray.single().jsonObject
            assertEquals(value, trial.getValue("values").jsonObject.getValue("flag").jsonPrimitive.boolean)
            val id = trial.getValue("id").jsonPrimitive.long
            assertEquals(HttpStatusCode.OK, client.send("/v1/optimizers/${optimizer.id}/trials/$id/complete", """{"reward":1}""").status)
        }
        assertEquals(2, client.get("/v1/optimizers/${optimizer.id}").objectBody().getValue("observationCount").jsonPrimitive.int)
    }

    @Test
    fun `failed and cancelled trials release reservations without inventing evidence`() = testApplication {
        application { resourceApi() }
        val optimizer = client.optimizer(DecisionSpaceDef(variables = mapOf("flag" to BoolSpec)))
        val trials = client.suggest(optimizer.id, 2)
        val first = trials[0].jsonObject.getValue("id").jsonPrimitive.long
        val second = trials[1].jsonObject.getValue("id").jsonPrimitive.long
        assertEquals(HttpStatusCode.OK, client.send("/v1/optimizers/${optimizer.id}/trials/$first/fail", """{"reason":"evaluation failed"}""").status)
        assertEquals(HttpStatusCode.OK, client.post("/v1/optimizers/${optimizer.id}/trials/$second/cancel").status)
        val description = client.get("/v1/optimizers/${optimizer.id}").objectBody()
        assertEquals(0, description.getValue("observationCount").jsonPrimitive.int)
        assertEquals(0, description.getValue("pendingCount").jsonPrimitive.int)
        assertEquals(2, client.suggest(optimizer.id, 2).size)
    }

    @Test
    fun `resource deletion releases retained capacity`() = testApplication {
        application { resourceApi(ServerState(capacity = 1)) }
        val first = client.create("stats", """{"sum":{}}""")
        val rejected = client.send("/v1/stats", """{"sum":{}}""")
        assertEquals(HttpStatusCode.ServiceUnavailable, rejected.status)
        assertEquals("1", rejected.headers[HttpHeaders.RetryAfter])
        assertEquals(HttpStatusCode.NoContent, client.delete("/v1/stats/${first.id}").status)
        assertEquals(HttpStatusCode.NotFound, client.get("/v1/stats/${first.id}").status)
        assertNotNull(client.create("stats", """{"sum":{}}""").id)
    }

    @Test
    fun `invalid input is rejected directly without creating partial resources`() = testApplication {
        application { resourceApi() }
        val bodies = listOf("{", """{"type":"unknown"}""", """{"typo":1}""")
        for (body in bodies) assertEquals(HttpStatusCode.BadRequest, client.send("/v1/stats", body).status)
        val badSchema = wireJson.encodeToString(CreateProblem(SchemaDef(mapOf("invalid" to NamedConstraint(BoolRef("missing"))))))
        assertEquals(HttpStatusCode.BadRequest, client.send("/v1/problems", badSchema).status)
        assertEquals("[]", client.get("/v1/problems").bodyAsText())
        assertEquals("[]", client.get("/v1/stats").bodyAsText())
        assertNotNull(client.create("stats", """{"sum":{}}"""))
    }

    @Test
    fun `unknown resources and trials return not found`() = testApplication {
        application { resourceApi() }
        for (collection in listOf("problems", "stats", "optimizers")) {
            assertEquals(HttpStatusCode.NotFound, client.get("/v1/$collection/missing").status)
            assertEquals(HttpStatusCode.NotFound, client.delete("/v1/$collection/missing").status)
        }
        val optimizer = client.optimizer(DecisionSpaceDef(variables = mapOf("flag" to BoolSpec)))
        assertEquals(HttpStatusCode.NotFound, client.send("/v1/optimizers/${optimizer.id}/trials/999/complete", """{"reward":1}""").status)
        assertEquals(HttpStatusCode.NotFound, client.get("/v1/jobs").status)
    }

    @Test
    fun `waiting queries respect their deadline without deleting the problem`() = testApplication {
        val state = ServerState()
        application { resourceApi(state) }
        val problem = client.create("problems", wireJson.encodeToString(CreateProblem(SchemaDef(mapOf("flag" to BoolSpec)))))
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        coroutineScope {
            val holder = async(start = CoroutineStart.UNDISPATCHED) {
                state.problems.access(problem.id) { _, _ ->
                    started.countDown()
                    assertTrue(release.await(5, TimeUnit.SECONDS))
                }
            }
            try {
                assertTrue(started.await(5, TimeUnit.SECONDS))
                val expired = client.send("/v1/problems/${problem.id}/solve", """{"timeoutMillis":20}""")
                assertEquals(HttpStatusCode.RequestTimeout, expired.status, expired.bodyAsText())
                assertEquals("operation_timed_out", expired.objectBody().getValue("code").jsonPrimitive.content)
            } finally {
                release.countDown()
            }
            holder.await()
        }
        assertEquals(HttpStatusCode.OK, client.send("/v1/problems/${problem.id}/solve", "{}").status)
    }

    @Test
    fun `large request bodies are rejected and health stays available`() = testApplication {
        application { resourceApi() }
        val response = client.send("/v1/stats", " ".repeat(2 * 1024 * 1024 + 1))
        assertEquals(HttpStatusCode.PayloadTooLarge, response.status)
        assertEquals(HttpStatusCode.OK, client.get("/health").status)
    }
}

internal suspend fun HttpClient.send(path: String, body: String): HttpResponse = post(path) {
    contentType(ContentType.Application.Json)
    setBody(body)
}

internal suspend fun HttpClient.create(collection: String, body: String = "{}"): ResourceInfo {
    val response = send("/v1/$collection", body)
    assertEquals(HttpStatusCode.Created, response.status, response.bodyAsText())
    val body = response.objectBody()
    val resource = ResourceInfo(body.getValue("id").jsonPrimitive.content, body["type"]?.jsonPrimitive?.content.orEmpty())
    assertEquals("/v1/${collection.substringBefore('?')}/${resource.id}", response.headers[HttpHeaders.Location])
    return resource
}

internal suspend fun HttpClient.optimizer(space: DecisionSpaceDef) = create("optimizers", wireJson.encodeToString(CreateOptimizer(space)))
internal suspend fun HttpClient.suggest(id: String, count: Int): JsonArray {
    val response = send("/v1/optimizers/$id/suggestions", """{"count":$count}""")
    assertEquals(HttpStatusCode.OK, response.status, response.bodyAsText())
    return response.objectBody().getValue("trials").jsonArray
}
internal suspend fun HttpResponse.objectBody(): JsonObject = wireJson.parseToJsonElement(bodyAsText()).jsonObject
internal suspend fun HttpResponse.arrayBody(): JsonArray = wireJson.parseToJsonElement(bodyAsText()).jsonArray
