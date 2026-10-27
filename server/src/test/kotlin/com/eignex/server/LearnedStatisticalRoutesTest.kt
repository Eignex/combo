package com.eignex.server

import io.ktor.client.request.get
import io.ktor.http.HttpStatusCode
import io.ktor.server.testing.testApplication
import kotlinx.serialization.json.*
import kotlin.test.*

class LearnedStatisticalRoutesTest {
    private val definition = """
        {"space":{"variables":{"workers":{"int":[1,8]}},
          "context":{"load":{"int":[1,8]}},
          "constraints":{"capacity":{"ge":[{"ref":"workers"},{"ref":"load"}]}},
          "interactions":[{"name":"workers_load",
            "lhs":{"kind":"IntContext","path":["context","load"]},
            "rhs":{"kind":"IntDecision","path":["workers"]}}]},
         "outcome":"reward","objective":{"maximize":"reward"},
         "statistics":{"demand":"mean"},
         "inputs":{"load":{"statistic":"demand","value":"mean"}},
         "learning":{"regression":{"bayesianRegression":{"featureSize":3}},
           "posterior":"multivariateGaussian"},"seed":42}
    """.trimIndent()

    @Test
    fun `linear proposals freeze statistical context and feedback preserves that context`() = testApplication {
        application { resourceApi() }
        val model = client.model(definition)
        val base = "/v1/models/${model.id}"
        assertEquals(HttpStatusCode.Conflict, client.send("$base/propose", "{}").status)
        assertEquals(HttpStatusCode.OK, client.send("$base/observe", """
            {"statistic":"demand","observation":{"value":2}}
        """.trimIndent()).status)
        val original = client.proposals(model.id, """{"count":3}""").map { it.jsonObject }
        assertEquals(3, original.size)
        assertEquals(3, original.map { it.getValue("values") }.distinct().size)
        for (proposal in original) {
            assertEquals(2, proposal.getValue("inputs").jsonObject.getValue("load.mean").jsonPrimitive.int)
            assertEquals(emptySet(), proposal.getValue("given").jsonObject.keys)
            assertEquals(setOf("workers"), proposal.getValue("values").jsonObject.keys)
            assertTrue(proposal.getValue("values").jsonObject.getValue("workers").jsonPrimitive.int >= 2)
        }
        client.send("$base/observe", """{"statistic":"demand","observation":{"value":6}}""")
        val newer = client.proposals(model.id, "{}").single().jsonObject
        assertEquals(4, newer.getValue("inputs").jsonObject.getValue("load.mean").jsonPrimitive.int)
        for (proposal in original.reversed()) {
            val id = proposal.getValue("id").jsonPrimitive.long
            val workers = proposal.getValue("values").jsonObject.getValue("workers").jsonPrimitive.int
            assertEquals(HttpStatusCode.OK, client.send("$base/observe", """
                {"proposal":$id,"values":{"reward":${workers * 2}}}
            """.trimIndent()).status)
        }
        val state = client.get(base).stateBody()
        assertEquals(3, state.getValue("observations").jsonPrimitive.int)
        val retained = state.getValue("proposals").jsonArray.take(3)
        assertTrue(retained.all { it.jsonObject.getValue("inputs").jsonObject.getValue("load.mean").jsonPrimitive.int == 2 })
        val recommendation = client.send("$base/recommend", "{}").objectBody()
        assertEquals("estimated", recommendation.getValue("status").jsonPrimitive.content)
        assertEquals(4, recommendation.getValue("inputs").jsonObject.getValue("load.mean").jsonPrimitive.int)
        assertTrue(recommendation.getValue("values").jsonObject.getValue("workers").jsonPrimitive.int >= 4)
        val infeasible = client.send("$base/evaluate", """{"values":{"workers":3}}""").objectBody()
        assertEquals("infeasible", infeasible.getValue("status").jsonPrimitive.content)
        for (operation in listOf("recommend", "propose", "evaluate")) {
            val values = if (operation == "evaluate") ",\"values\":{\"workers\":8}" else ""
            assertEquals(HttpStatusCode.BadRequest, client.send("$base/$operation", """{"given":{"load":1}$values}""").status)
        }
        val first = original.first().getValue("id").jsonPrimitive.long
        assertEquals(HttpStatusCode.Conflict, client.send("$base/observe", """
            {"proposal":$first,"values":{"reward":1}}
        """.trimIndent()).status)
    }

    @Test
    fun `statistical contexts also work with the default learner`() = testApplication {
        application { resourceApi() }
        val json = wireJson.parseToJsonElement(definition).jsonObject
        val model = client.model(JsonObject(json - "learning").toString())
        val base = "/v1/models/${model.id}"
        client.send("$base/observe", """{"statistic":"demand","observation":{"value":3}}""")
        val proposal = client.proposals(model.id, "{}").single().jsonObject
        val id = proposal.getValue("id").jsonPrimitive.long
        assertEquals(3, proposal.getValue("inputs").jsonObject.getValue("load.mean").jsonPrimitive.int)
        assertEquals(HttpStatusCode.OK, client.send("$base/observe", """
            {"proposal":$id,"values":{"reward":10}}
        """.trimIndent()).status)
        assertEquals("estimated", client.send("$base/recommend", "{}").objectBody().getValue("status").jsonPrimitive.content)
    }

    @Test
    fun `fractional or out of domain inputs fail without issuing proposals`() = testApplication {
        application { resourceApi() }
        for (value in listOf(2.5, 9.0)) {
            val model = client.model(definition)
            val base = "/v1/models/${model.id}"
            client.send("$base/observe", """{"statistic":"demand","observation":{"value":$value}}""")
            assertEquals(HttpStatusCode.BadRequest, client.send("$base/propose", "{}").status)
            assertTrue(client.get(base).stateBody().getValue("proposals").jsonArray.isEmpty())
        }
    }
}
