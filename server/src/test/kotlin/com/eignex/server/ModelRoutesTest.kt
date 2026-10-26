package com.eignex.server

import io.ktor.client.HttpClient
import io.ktor.client.request.*
import io.ktor.client.statement.*
import io.ktor.http.*
import io.ktor.server.testing.testApplication
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.serialization.json.*
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.test.*

class ModelRoutesTest {
    @Test
    fun `known objectives recommend proven optima under each request context`() = testApplication {
        application { resourceApi() }
        val model = client.model("""
            {"variables":{
              "workers":{"type":"integer","min":1,"max":8},
              "load":{"type":"integer","role":"context","min":1,"max":8}},
             "constraints":[{"terms":{"workers":1,"load":-1},"atLeast":0}],
             "objective":{"minimize":"workers"}}
        """.trimIndent())
        for (load in listOf(3, 6, 2)) {
            val response = client.send("/v1/models/${model.id}/recommend", """{"given":{"load":$load}}""")
            assertEquals(HttpStatusCode.OK, response.status, response.bodyAsText())
            val result = response.objectBody()
            assertEquals("optimal", result.getValue("status").jsonPrimitive.content)
            assertEquals(load, result.getValue("values").jsonObject.getValue("workers").jsonPrimitive.int)
            assertEquals(load, result.getValue("given").jsonObject.getValue("load").jsonPrimitive.int)
            assertEquals(load, result.getValue("objective").jsonObject.getValue("value").jsonPrimitive.int)
        }
        val description = client.get("/v1/models/${model.id}").objectBody()
        assertEquals(setOf("evaluate", "recommend"), description.getValue("capabilities").jsonArray.map { it.jsonPrimitive.content }.toSet())
        for (name in listOf("klause", "kumulant", "combo")) assertFalse(name in description.toString())
    }

    @Test
    fun `known objectives can combine variables through linear relationships`() = testApplication {
        application { resourceApi() }
        val model = client.model("""
            {"variables":{"workers":{"type":"integer","min":1,"max":8},"cost":{"type":"integer","min":0,"max":20}},
             "constraints":[{"terms":{"cost":1,"workers":-2},"equal":0},{"terms":{"workers":1},"atLeast":3}],
             "objective":{"minimize":"cost"}}
        """.trimIndent())
        val response = client.send("/v1/models/${model.id}/recommend", "{}")
        assertEquals(HttpStatusCode.OK, response.status, response.bodyAsText())
        assertEquals(6, response.objectBody().getValue("objective").jsonObject.getValue("value").jsonPrimitive.int)
        assertEquals(3, response.objectBody().getValue("values").jsonObject.getValue("workers").jsonPrimitive.int)
    }

    @Test
    fun `maximization and evaluation use the same variable definitions`() = testApplication {
        application { resourceApi() }
        val model = client.model("""
            {"variables":{"workers":{"type":"integer","min":1,"max":8}},
             "constraints":[{"terms":{"workers":1},"atMost":4}],"objective":{"maximize":"workers"}}
        """.trimIndent())
        val response = client.send("/v1/models/${model.id}/recommend", "{}")
        assertEquals(HttpStatusCode.OK, response.status, response.bodyAsText())
        assertEquals(4, response.objectBody().getValue("values").jsonObject.getValue("workers").jsonPrimitive.int)
        val valid = client.send("/v1/models/${model.id}/evaluate", """{"values":{"workers":3}}""")
        assertEquals("feasible", valid.objectBody().getValue("status").jsonPrimitive.content)
        assertEquals(3, valid.objectBody().getValue("objective").jsonObject.getValue("value").jsonPrimitive.int)
        val invalid = client.send("/v1/models/${model.id}/evaluate", """{"values":{"workers":5}}""")
        assertEquals("infeasible", invalid.objectBody().getValue("status").jsonPrimitive.content)
    }

    @Test
    fun `infeasible models return a verdict rather than a transport failure`() = testApplication {
        application { resourceApi() }
        val model = client.model("""
            {"variables":{"flag":{"type":"boolean"}},
             "constraints":[{"terms":{"flag":1},"equal":1},{"terms":{"flag":1},"equal":0}]}
        """.trimIndent())
        val response = client.send("/v1/models/${model.id}/recommend", "{}")
        assertEquals(HttpStatusCode.OK, response.status, response.bodyAsText())
        assertEquals("infeasible", response.objectBody().getValue("status").jsonPrimitive.content)
    }

    @Test
    fun `observed objectives propose experiments and learn from named outcomes`() = testApplication {
        application { resourceApi() }
        val model = client.model("""
            {"variables":{"workers":{"type":"integer","min":0,"max":3},"latency":{"role":"observed"}},
             "objective":{"minimize":"latency"},"stat":{"ewma":{"alpha":0.1}}}
        """.trimIndent())
        val early = client.send("/v1/models/${model.id}/recommend", "{}")
        assertEquals(HttpStatusCode.Conflict, early.status)
        assertEquals("insufficient_evidence", early.objectBody().getValue("code").jsonPrimitive.content)
        val proposals = client.proposals(model.id, """{"count":4}""")
        assertEquals(4, proposals.size)
        assertEquals(4, proposals.map { it.jsonObject.getValue("values") }.distinct().size)
        val reserved = client.get("/v1/models/${model.id}").stateBody()
        assertEquals(0, reserved.getValue("observations").jsonPrimitive.int)
        assertEquals(0.0, reserved.getValue("stat").jsonObject.getValue("totalWeights").jsonPrimitive.double)
        for (proposal in proposals.reversed()) {
            val point = proposal.jsonObject
            val id = point.getValue("id").jsonPrimitive.long
            val workers = point.getValue("values").jsonObject.getValue("workers").jsonPrimitive.int
            val response = client.send("/v1/models/${model.id}/observe", """{"proposal":$id,"values":{"latency":${10 - workers}}}""")
            assertEquals(HttpStatusCode.OK, response.status, response.bodyAsText())
        }
        val state = client.get("/v1/models/${model.id}").stateBody()
        assertEquals(4, state.getValue("observations").jsonPrimitive.int)
        assertEquals(4.0, state.getValue("stat").jsonObject.getValue("totalWeights").jsonPrimitive.double)
        val recommended = client.send("/v1/models/${model.id}/recommend", "{}")
        assertEquals(HttpStatusCode.OK, recommended.status, recommended.bodyAsText())
        assertEquals("estimated", recommended.objectBody().getValue("status").jsonPrimitive.content)
        assertTrue("estimate" in recommended.objectBody().getValue("objective").jsonObject)
        assertFalse("value" in recommended.objectBody().getValue("objective").jsonObject)
        val evaluated = client.send("/v1/models/${model.id}/evaluate", """{"values":{"workers":2}}""")
        assertEquals(HttpStatusCode.OK, evaluated.status, evaluated.bodyAsText())
        assertEquals("estimated", evaluated.objectBody().getValue("status").jsonPrimitive.content)
        val first = proposals.first().jsonObject.getValue("id").jsonPrimitive.long
        val repeated = client.send("/v1/models/${model.id}/observe", """{"proposal":$first,"values":{"latency":1}}""")
        assertEquals(HttpStatusCode.Conflict, repeated.status)
        assertEquals(state, client.get("/v1/models/${model.id}").stateBody())
    }

    @Test
    fun `learned minimization and maximization recommend opposite decisions`() = testApplication {
        application { resourceApi() }
        for ((direction, expected) in listOf("minimize" to false, "maximize" to true)) {
            val model = client.model("""
                {"variables":{"enabled":{"type":"boolean"},"quality":{"role":"observed"}},"objective":{"$direction":"quality"}}
            """.trimIndent())
            for (proposal in client.proposals(model.id, """{"count":2}""")) {
                val point = proposal.jsonObject
                val id = point.getValue("id").jsonPrimitive.long
                val reward = if (point.getValue("values").jsonObject.getValue("enabled").jsonPrimitive.boolean) 10 else 0
                val response = client.send("/v1/models/${model.id}/observe", """{"proposal":$id,"values":{"quality":$reward}}""")
                assertEquals(HttpStatusCode.OK, response.status, response.bodyAsText())
            }
            val result = client.send("/v1/models/${model.id}/recommend", "{}")
            assertEquals(HttpStatusCode.OK, result.status, result.bodyAsText())
            assertEquals(expected, result.objectBody().getValue("values").jsonObject.getValue("enabled").jsonPrimitive.boolean)
        }
    }

    @Test
    fun `each proposal retains its own hard context for later feedback`() = testApplication {
        application { resourceApi() }
        val model = client.model("""
            {"variables":{"enabled":{"type":"boolean"},"premium":{"type":"boolean","role":"context"},"quality":{"role":"observed"}},
             "constraints":[{"terms":{"enabled":1,"premium":-1},"equal":0}],"objective":{"maximize":"quality"}}
        """.trimIndent())
        for (premium in listOf(true, false)) {
            val point = client.proposals(model.id, """{"given":{"premium":$premium}}""").single().jsonObject
            assertEquals(premium, point.getValue("values").jsonObject.getValue("enabled").jsonPrimitive.boolean)
            assertEquals(premium, point.getValue("given").jsonObject.getValue("premium").jsonPrimitive.boolean)
            val id = point.getValue("id").jsonPrimitive.long
            val response = client.send("/v1/models/${model.id}/observe", """{"proposal":$id,"values":{"quality":1}}""")
            assertEquals(HttpStatusCode.OK, response.status, response.bodyAsText())
        }
        val result = client.send("/v1/models/${model.id}/recommend", """{"given":{"premium":false}}""")
        assertEquals(HttpStatusCode.OK, result.status, result.bodyAsText())
        assertFalse(result.objectBody().getValue("values").jsonObject.getValue("enabled").jsonPrimitive.boolean)
    }

    @Test
    fun `real valued proposals retain assignments through observation and evaluation`() = testApplication {
        application { resourceApi() }
        val model = client.model("""
            {"variables":{"rate":{"min":0.1,"max":0.9},"quality":{"role":"observed"}},"objective":{"maximize":"quality"}}
        """.trimIndent())
        val point = client.proposals(model.id).single().jsonObject
        val id = point.getValue("id").jsonPrimitive.long
        val input = point.getValue("values")
        assertEquals(HttpStatusCode.OK, client.send("/v1/models/${model.id}/observe", """{"proposal":$id,"values":{"quality":0.7}}""").status)
        val result = client.send("/v1/models/${model.id}/evaluate", """{"values":$input}""")
        assertEquals(HttpStatusCode.OK, result.status, result.bodyAsText())
        assertEquals(input, result.objectBody().getValue("values"))
    }

    @Test
    fun `cancelling a proposal releases its reservation without an outcome`() = testApplication {
        application { resourceApi() }
        val model = client.model("""
            {"variables":{"enabled":{"type":"boolean"},"quality":{"role":"observed"}},"objective":{"maximize":"quality"}}
        """.trimIndent())
        val points = client.proposals(model.id, """{"count":2}""")
        assertTrue(client.proposals(model.id).isEmpty())
        val point = points.first().jsonObject
        val id = point.getValue("id").jsonPrimitive.long
        assertEquals(HttpStatusCode.OK, client.delete("/v1/models/${model.id}/proposals/$id").status)
        val next = client.proposals(model.id).single().jsonObject
        assertEquals(point.getValue("values"), next.getValue("values"))
        assertTrue(next.getValue("id") != point.getValue("id"))
        assertEquals(0, client.get("/v1/models/${model.id}").stateBody().getValue("observations").jsonPrimitive.int)
        assertEquals(HttpStatusCode.Conflict, client.send("/v1/models/${model.id}/observe", """{"proposal":$id,"values":{"quality":1}}""").status)
    }

    @Test
    fun `invalid definitions and observations do not create evidence`() = testApplication {
        application { resourceApi() }
        for (body in listOf(
            "{}", "[]", """{"engine":"klause"}""",
            """{"variables":{"x":{"type":"integer","min":1.5,"max":3}}}""",
            """{"variables":{"x":{"type":"integer","min":1,"max":3}},"objective":{"minimize":"missing"}}""",
            """{"variables":{"x":{"type":"integer","min":1,"max":3}},"constraints":[{"terms":{"unknown":1},"equal":1}]}""",
            """{"variables":{"outcome":{"role":"observed"}},"objective":{"minimize":"outcome"}}""",
        )) assertEquals(HttpStatusCode.BadRequest, client.send("/v1/models", body).status, body)
        assertEquals("[]", client.get("/v1/models").bodyAsText())
        val model = client.model("""
            {"variables":{"enabled":{"type":"boolean"},"quality":{"role":"observed"}},"objective":{"maximize":"quality"}}
        """.trimIndent())
        val id = client.proposals(model.id).single().jsonObject.getValue("id").jsonPrimitive.long
        for (body in listOf(
            """{"proposal":$id,"values":{"other":1}}""",
            """{"proposal":$id,"values":{"quality":"1"}}""",
            """{"proposal":$id,"values":{"quality":1},"weight":0}""",
        )) assertEquals(HttpStatusCode.BadRequest, client.send("/v1/models/${model.id}/observe", body).status)
        assertEquals(0, client.get("/v1/models/${model.id}").stateBody().getValue("observations").jsonPrimitive.int)
    }

    @Test
    fun `model deletion releases capacity and unknown operations return direct errors`() = testApplication {
        application { resourceApi(ServerState(capacity = 1)) }
        val model = client.stat("""{"sum":{}}""")
        assertEquals(HttpStatusCode.ServiceUnavailable, client.send("/v1/models", """{"stat":{"sum":{}}}""").status)
        assertEquals(HttpStatusCode.BadRequest, client.send("/v1/models/${model.id}/recommend", "{}").status)
        assertEquals(HttpStatusCode.NoContent, client.delete("/v1/models/${model.id}").status)
        assertEquals(HttpStatusCode.NotFound, client.get("/v1/models/${model.id}").status)
        assertNotNull(client.stat("""{"sum":{}}"""))
        for (path in listOf("/v1/stats", "/v1/problems", "/v1/optimizers")) assertEquals(HttpStatusCode.NotFound, client.get(path).status)
    }

    @Test
    fun `waiting recommendations respect deadlines without deleting their model`() = testApplication {
        val state = ServerState()
        application { resourceApi(state) }
        val model = client.model("""{"variables":{"enabled":{"type":"boolean"}}}""")
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        coroutineScope {
            val holder = async(start = CoroutineStart.UNDISPATCHED) {
                state.models.access(model.id) { _, _ -> started.countDown(); assertTrue(release.await(5, TimeUnit.SECONDS)) }
            }
            try {
                assertTrue(started.await(5, TimeUnit.SECONDS))
                val response = client.send("/v1/models/${model.id}/recommend", """{"timeoutMillis":20}""")
                assertEquals(HttpStatusCode.RequestTimeout, response.status, response.bodyAsText())
            } finally { release.countDown() }
            holder.await()
        }
        assertEquals(HttpStatusCode.OK, client.send("/v1/models/${model.id}/recommend", "{}").status)
    }

    @Test
    fun `large bodies are rejected and health remains available`() = testApplication {
        application { resourceApi() }
        assertEquals(HttpStatusCode.PayloadTooLarge, client.send("/v1/models", " ".repeat(2 * 1024 * 1024 + 1)).status)
        assertEquals(HttpStatusCode.OK, client.get("/health").status)
    }
}

internal suspend fun HttpClient.send(path: String, body: String): HttpResponse = post(path) {
    contentType(ContentType.Application.Json)
    setBody(body)
}
internal suspend fun HttpClient.model(body: String): ResourceInfo {
    val response = send("/v1/models", body)
    assertEquals(HttpStatusCode.Created, response.status, response.bodyAsText())
    val id = response.objectBody().getValue("id").jsonPrimitive.content
    assertEquals("/v1/models/$id", response.headers[HttpHeaders.Location])
    return ResourceInfo(id, "model")
}
internal suspend fun HttpClient.stat(body: String, seed: Int = 0): ResourceInfo = model("""{"stat":$body,"seed":$seed}""")
internal suspend fun HttpClient.proposals(id: String, body: String = "{}"): JsonArray {
    val response = send("/v1/models/$id/propose", body)
    assertEquals(HttpStatusCode.OK, response.status, response.bodyAsText())
    return response.objectBody().getValue("proposals").jsonArray
}
internal suspend fun HttpResponse.objectBody(): JsonObject = wireJson.parseToJsonElement(bodyAsText()).jsonObject
internal suspend fun HttpResponse.stateBody(): JsonObject = objectBody().getValue("state").jsonObject
