package com.eignex.server

import io.ktor.client.request.get
import io.ktor.http.HttpStatusCode
import io.ktor.server.testing.testApplication
import kotlinx.serialization.json.*
import kotlin.test.*

class RealContextRoutesTest {
    @Test
    fun `continuous context constrains real decisions without leaking between requests`() = testApplication {
        application { resourceApi() }
        val model = client.model("""
            {"space":{"variables":{"capacity":{"float":{"min":0.0,"max":10.0}}},
              "context":{"load":{"float":{"min":0.0,"max":10.0}}},
              "constraints":{"enough":{"floatLinear":{
                "coeffs":[1.0,-1.0],"varNames":["capacity","load"],"op":"GE","bound":0.0}}}},
              "objective":{"minimize":"capacity"}}
        """.trimIndent())
        val base = "/v1/models/${model.id}"
        for (load in listOf(2.5, 7.25, 1.125)) {
            val response = client.send("$base/recommend", """{"given":{"load":$load}}""")
            assertEquals(HttpStatusCode.OK, response.status)
            val body = response.objectBody()
            assertEquals(load, body.getValue("given").jsonObject.getValue("load").jsonPrimitive.double)
            assertEquals(load, body.getValue("values").jsonObject.getValue("capacity").jsonPrimitive.double, 1e-10)
            val invalid = client.send("$base/evaluate", """
                {"given":{"load":$load},"values":{"capacity":${load - 0.125}}}
            """.trimIndent())
            assertEquals("infeasible", invalid.objectBody().getValue("status").jsonPrimitive.content)
        }
    }

    @Test
    fun `fractional statistics feed both persistent linear and default learned models`() = testApplication {
        application { resourceApi() }
        for (linear in listOf(true, false)) {
            val learning = if (linear) """,
                "learning":{"regression":{"bayesianRegression":{"featureSize":3}},"posterior":"multivariateGaussian"}
            """.trimIndent() else ""
            val model = client.model("""
                {"space":{"variables":{"workers":{"float":{"min":1.0,"max":8.0}}},
                  "context":{"load":{"float":{"min":0.0,"max":8.0}}},
                  "constraints":{"capacity":{"floatLinear":{
                    "coeffs":[1.0,-1.0],"varNames":["workers","load"],"op":"GE","bound":0.0}}},
                  "interactions":[{"name":"workers_load",
                    "lhs":{"kind":"FloatContext","path":["context","load"]},
                    "rhs":{"kind":"FloatDecision","path":["workers"]}}]},
                 "outcome":"reward","objective":{"maximize":"reward"},
                 "statistics":{"demand":"mean"},
                 "inputs":{"load":{"statistic":"demand","value":"mean"}}$learning}
            """.trimIndent())
            val base = "/v1/models/${model.id}"
            client.send("$base/observe", """{"statistic":"demand","observation":{"value":2.5}}""")
            val proposals = client.proposals(model.id, """{"count":2}""").map { it.jsonObject }
            assertEquals(2, proposals.size)
            assertTrue(proposals.all {
                it.getValue("inputs").jsonObject.getValue("load.mean").jsonPrimitive.double == 2.5
            })
            client.send("$base/observe", """{"statistic":"demand","observation":{"value":6.5}}""")
            for (proposal in proposals.reversed()) {
                val id = proposal.getValue("id").jsonPrimitive.long
                assertEquals(HttpStatusCode.OK, client.send("$base/observe", """
                    {"proposal":$id,"values":{"reward":10}}
                """.trimIndent()).status)
            }
            val recommendation = client.send("$base/recommend", "{}")
            assertEquals(HttpStatusCode.OK, recommendation.status)
            val body = recommendation.objectBody()
            assertEquals(4.5, body.getValue("inputs").jsonObject.getValue("load.mean").jsonPrimitive.double)
            assertTrue(body.getValue("values").jsonObject.getValue("workers").jsonPrimitive.double >= 4.5)
            val state = client.get(base).stateBody()
            assertEquals(2, state.getValue("observations").jsonPrimitive.int)
            assertTrue(state.getValue("proposals").jsonArray.all {
                it.jsonObject.getValue("inputs").jsonObject.getValue("load.mean").jsonPrimitive.double == 2.5
            })
        }
    }
}
