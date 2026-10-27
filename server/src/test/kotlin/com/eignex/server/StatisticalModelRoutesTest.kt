package com.eignex.server

import com.eignex.kumulant.stat.decay.EwmaMeanStat

import io.ktor.client.request.get
import io.ktor.http.HttpStatusCode
import io.ktor.server.testing.testApplication
import kotlinx.serialization.json.*
import kotlin.test.*

class StatisticalModelRoutesTest {
    @Test
    fun `observed statistics constrain native Klause decisions without becoming decisions themselves`() = testApplication {
        application { resourceApi() }
        val model = client.model("""
            {"space":{"variables":{"capacity":{"int":[0,200]}},
              "constraints":{"enough":{"ge":[{"ref":"capacity"},{"ref":"latency"}]}}},
             "objective":{"minimize":"capacity"},
             "statistics":{"timings":{"band":{"inner":"variance","k":1}}},
             "inputs":{"latency":{"statistic":"timings","projections":{"mean":["center"]}}}}
        """.trimIndent())
        assertEquals(HttpStatusCode.Conflict, client.send("/v1/models/${model.id}/recommend", "{}").status)
        for (value in listOf(80, 120)) {
            assertEquals(HttpStatusCode.OK, client.send("/v1/models/${model.id}/observe", """
                {"statistic":"timings","observation":{"value":$value}}
            """.trimIndent()).status)
        }
        val result = client.send("/v1/models/${model.id}/recommend", "{}")
        assertEquals(HttpStatusCode.OK, result.status)
        val body = result.objectBody()
        assertEquals(setOf("capacity"), body.getValue("values").jsonObject.keys)
        assertEquals(120, body.getValue("values").jsonObject.getValue("capacity").jsonPrimitive.int)
        assertEquals(80.0, body.getValue("inputs").jsonObject.getValue("latency.lower").jsonPrimitive.double)
        assertEquals(120.0, body.getValue("inputs").jsonObject.getValue("latency.upper").jsonPrimitive.double)
        val invalid = client.send("/v1/models/${model.id}/evaluate", """{"values":{"capacity":100}}""")
        assertEquals("infeasible", invalid.objectBody().getValue("status").jsonPrimitive.content)
        val override = client.send("/v1/models/${model.id}/evaluate", """{"values":{"capacity":100,"latency":80}}""")
        assertEquals(HttpStatusCode.BadRequest, override.status)
    }

    @Test
    fun `fractional scalar statistics can be checked in a native real constraint`() = testApplication {
        application { resourceApi() }
        val model = client.model("""
            {"space":{"constraints":{"limit":{"floatLinear":{
              "coeffs":[1.0],"varNames":["latency"],"op":"LT","bound":100.0}}}},
             "statistics":{"latency":{"ewma":{"alpha":1.0}}},
             "inputs":{"latency":{"statistic":"latency","value":"mean"}}}
        """.trimIndent())
        val native = EwmaMeanStat(1.0)
        for ((value, status) in listOf(80.5 to "feasible", 120.5 to "infeasible")) {
            native.update(value)
            assertEquals(HttpStatusCode.OK, client.send("/v1/models/${model.id}/observe", """
                {"statistic":"latency","observation":{"value":$value}}
            """.trimIndent()).status)
            val result = client.send("/v1/models/${model.id}/recommend", "{}")
            assertEquals(HttpStatusCode.OK, result.status)
            assertEquals(status, result.objectBody().getValue("status").jsonPrimitive.content)
            val actual = result.objectBody().getValue("inputs").jsonObject.getValue("latency.mean").jsonPrimitive.double
            assertEquals(native.read().mean, actual, 1e-10)
        }
    }

    @Test
    fun `named native statistics also work without a decision space`() = testApplication {
        application { resourceApi() }
        val model = client.model("""{"statistics":{"latency":{"ewma":{"alpha":0.1}}}}""")
        val response = client.send("/v1/models/${model.id}/observe", """
            {"statistic":"latency","observation":{"value":42}}
        """.trimIndent())
        assertEquals(HttpStatusCode.OK, response.status)
        val state = client.get("/v1/models/${model.id}").stateBody()
        assertEquals(42.0, state.getValue("statistics").jsonObject.getValue("latency").jsonObject.getValue("mean").jsonPrimitive.double)
    }
}
