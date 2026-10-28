package com.eignex.server

import io.ktor.client.request.get
import io.ktor.http.HttpStatusCode
import io.ktor.server.testing.testApplication
import kotlinx.serialization.json.*
import kotlin.test.*

class NativeProblemRoutesTest {
    @Test
    fun `irrational solver results round trip through evaluation and exact given values`() = testApplication {
        application { resourceApi() }
        val model = client.model("""
            {"problem":{"reals":[{"lower":0,"upper":2},{"lower":2,"upper":2}],
             "factors":[{"power":{"base":0,"exponent":2,"result":1,"real":true}}]}}
        """.trimIndent())
        val base = "/v1/models/${model.id}"
        val recommendation = client.send("$base/recommend", "{}").objectBody()
        assertEquals("feasible", recommendation.getValue("status").jsonPrimitive.content)
        val values = recommendation.getValue("values").jsonObject
        val exact = values.getValue("reals.0").jsonObject.getValue("exact")
        assertTrue("root" in exact.jsonObject)
        val checked = client.send("$base/evaluate", """{"values":$values}""").objectBody()
        assertEquals("feasible", checked.getValue("status").jsonPrimitive.content)
        val given = client.send("$base/recommend", """{"given":{"reals.0":{"exact":$exact,"approximation":999}}}""").objectBody()
        assertEquals("feasible", given.getValue("status").jsonPrimitive.content)
        assertEquals(exact, given.getValue("given").jsonObject.getValue("reals.0").jsonObject["exact"])
        val rounded = client.send("$base/evaluate", """{"values":{"reals.0":1.4142135623730951,"reals.1":2}}""").objectBody()
        assertEquals("infeasible", rounded.getValue("status").jsonPrimitive.content)
    }

    @Test
    fun `symbolic results use the same native exact contract for values and given`() = testApplication {
        application { resourceApi() }
        val exact = """{"add":[{"pi":{}},{"exp":"1"}]}"""
        val model = client.model("""
            {"problem":{"reals":[{}],"factors":[{"exactRealEquality":{"real":0,"value":$exact}}]}}
        """.trimIndent())
        val base = "/v1/models/${model.id}"
        val recommended = client.send("$base/recommend", "{}").objectBody()
        assertEquals("feasible", recommended.getValue("status").jsonPrimitive.content)
        val values = recommended.getValue("values").jsonObject
        assertEquals(Json.parseToJsonElement(exact), values.getValue("reals.0").jsonObject["exact"])
        val checked = client.send("$base/evaluate", """{"values":$values}""").objectBody()
        assertEquals("feasible", checked.getValue("status").jsonPrimitive.content)
        assertEquals(values, checked["values"])
        val given = client.send("$base/evaluate", """{"values":{},"given":{"reals.0":{"exact":$exact}}}""").objectBody()
        assertEquals("feasible", given.getValue("status").jsonPrimitive.content)
        assertEquals(Json.parseToJsonElement(exact), given.getValue("given").jsonObject.getValue("reals.0").jsonObject["exact"])
    }

    @Test
    fun `malformed or approximation only real objects are rejected`() = testApplication {
        application { resourceApi() }
        val model = client.model("""{"problem":{"reals":[{}]}}""")
        for (value in listOf(
            """{"approximation":1.4}""", """{"exact":{"unknown":"1"}}""",
            """{"exact":{"root":{"coefficients":["-2","0","1"],"index":3}}}""",
            """{"exact":{"power":[{"pi":{}},-1]}}""", """{"exact":{"log":"0"}}""",
            """{"exact":"exp(1)"}""", """{"exact":{"ref":"reals.0"}}""",
            """{"exact":{"pi":{}},"other":1}""",
        )) {
            val response = client.send("/v1/models/${model.id}/evaluate", """{"values":{"reals.0":$value}}""")
            assertEquals(HttpStatusCode.BadRequest, response.status, value)
        }
    }

    @Test
    fun `learned proposals retain exact contexts through asynchronous feedback`() = testApplication {
        application { resourceApi() }
        for (learning in listOf("", """, "learning":{"regression":{"bayesianRegression":{"featureSize":2}},"posterior":"multivariateGaussian"}""")) {
            val model = client.model("""
                {"problem":{"bools":1,"reals":[{}]},"outcome":"reward",
                 "objective":{"maximize":"reward"},"seed":42$learning}
            """.trimIndent())
            val base = "/v1/models/${model.id}"
            val proposals = listOf("""{"exp":"1"}""", """{"pi":{}}""").map { exact ->
                val proposal = client.proposals(model.id, """{"given":{"reals.0":{"exact":$exact}}}""").single().jsonObject
                assertEquals(Json.parseToJsonElement(exact), proposal.getValue("given").jsonObject.getValue("reals.0").jsonObject["exact"])
                proposal
            }
            for (proposal in proposals.reversed()) {
                val id = proposal.getValue("id").jsonPrimitive.long
                assertEquals(HttpStatusCode.OK, client.send("$base/observe", """{"proposal":$id,"values":{"reward":10}}""").status)
                val evaluated = client.send("$base/evaluate", buildJsonObject {
                    put("values", proposal.getValue("values")); put("given", proposal.getValue("given"))
                }.toString()).objectBody()
                assertEquals("estimated", evaluated.getValue("status").jsonPrimitive.content)
                assertEquals(proposal["given"], evaluated["given"])
            }
            val retained = client.get(base).objectBody().getValue("state").jsonObject.getValue("proposals").jsonArray.map { it.jsonObject }
            assertEquals(proposals.map { it["given"] }, retained.map { it["given"] })
            val estimate = client.send("$base/recommend", buildJsonObject {
                put("given", proposals.first().getValue("given"))
            }.toString()).objectBody()
            assertEquals("estimated", estimate.getValue("status").jsonPrimitive.content)
            assertEquals(proposals.first()["given"], estimate["given"])
        }
    }

    @Test
    fun `native table and sparse Long domain use the existing lifecycle unchanged`() = testApplication {
        application { resourceApi() }
        val problem = """{"ints":[{"lower":3000000000,"upper":3000000010,"values":[[3000000000,3000000000],[3000000010,3000000010]]}],"factors":[{"table":{"vars":[0],"tuples":[[3000000010]]}}]}"""
        val model = client.model("""{"problem":$problem,"objective":{"minimize":"ints.0"}}""")
        val base = "/v1/models/${model.id}"
        val description = client.get(base).objectBody().getValue("definition").jsonObject
        assertEquals(
            com.eignex.klause.formats.json.ProblemJson.default.encodeToJsonElement(
                com.eignex.klause.formats.json.ProblemJson.default.decodeFromString(problem),
            ), description["problem"],
        )
        val recommendation = client.send("$base/recommend", "{}").objectBody()
        assertEquals("optimal", recommendation.getValue("status").jsonPrimitive.content)
        assertEquals(3000000010L, recommendation.getValue("values").jsonObject.getValue("ints.0").jsonPrimitive.long)
        val invalid = client.send("$base/evaluate", """{"values":{"ints.0":3000000000}}""").objectBody()
        assertEquals("infeasible", invalid.getValue("status").jsonPrimitive.content)
    }

    @Test
    fun `open real objectives remain unbounded and exact rational pins survive evaluation`() = testApplication {
        application { resourceApi() }
        val model = client.model("""{"problem":{"reals":[{}]},"objective":{"maximize":"reals.0"}}""")
        val base = "/v1/models/${model.id}"
        assertEquals("unbounded", client.send("$base/recommend", "{}").objectBody().getValue("status").jsonPrimitive.content)
        val pinned = client.send("$base/evaluate", """{"values":{"reals.0":"1/3"}}""").objectBody()
        assertEquals("feasible", pinned.getValue("status").jsonPrimitive.content)
        assertEquals("1/3", pinned.getValue("values").jsonObject.getValue("reals.0").jsonPrimitive.content)
    }

    @Test
    fun `native learned models share forest and linear feedback with pinned columns`() = testApplication {
        application { resourceApi() }
        for (learning in listOf("", """, "learning":{"regression":{"bayesianRegression":{"featureSize":2}},"posterior":"multivariateGaussian"}""")) {
            val model = client.model("""
                {"problem":{"bools":1,"ints":[{"lower":0,"upper":3}]},
                 "outcome":"reward","objective":{"maximize":"reward"},"seed":42$learning}
            """.trimIndent())
            val proposal = client.proposals(model.id, """{"given":{"ints.0":2}}""").single().jsonObject
            assertEquals(setOf("bools.0"), proposal.getValue("values").jsonObject.keys)
            assertEquals(2, proposal.getValue("given").jsonObject.getValue("ints.0").jsonPrimitive.int)
            val id = proposal.getValue("id").jsonPrimitive.long
            val observed = client.send("/v1/models/${model.id}/observe", """{"proposal":$id,"values":{"reward":10}}""")
            assertEquals(HttpStatusCode.OK, observed.status)
            assertEquals(1, observed.objectBody().getValue("observations").jsonPrimitive.int)
            val estimate = client.send("/v1/models/${model.id}/recommend", """{"given":{"ints.0":3}}""").objectBody()
            assertEquals("estimated", estimate.getValue("status").jsonPrimitive.content)
            assertEquals(3, estimate.getValue("given").jsonObject.getValue("ints.0").jsonPrimitive.int)
        }
    }

    @Test
    fun `statistics pin native columns for known and learned objectives`() = testApplication {
        application { resourceApi() }
        for (learned in listOf(false, true)) {
            val goal = if (learned) """"outcome":"reward","objective":{"maximize":"reward"}"""
                else """"objective":{"minimize":"ints.0"}"""
            val model = client.model("""
                {"problem":{"ints":[{"lower":0,"upper":8},{"lower":0,"upper":8}],
                  "factors":[{"linear":{"vars":[0,1],"coefficients":[1,-1],"op":"GE","bound":0}}]},
                 "statistics":{"demand":"mean"},"inputs":{"ints.1":{"statistic":"demand","value":"mean"}},$goal}
            """.trimIndent())
            val base = "/v1/models/${model.id}"
            val operation = if (learned) "propose" else "recommend"
            assertEquals(HttpStatusCode.Conflict, client.send("$base/$operation", "{}").status)
            assertEquals(HttpStatusCode.OK, client.send("$base/observe", """{"statistic":"demand","observation":{"value":3}}""").status)
            val selection = if (learned) client.proposals(model.id).single().jsonObject else client.send("$base/recommend", "{}").objectBody()
            assertEquals(setOf("ints.0"), selection.getValue("values").jsonObject.keys)
            assertTrue(selection.getValue("values").jsonObject.getValue("ints.0").jsonPrimitive.int >= 3)
            assertEquals(3, selection.getValue("inputs").jsonObject.getValue("ints.1.mean").jsonPrimitive.int)
        }
    }

    @Test
    fun `arbitrary width integer witnesses round trip through evaluation`() = testApplication {
        application { resourceApi() }
        val number = "18446744073709551616"
        val model = client.model("""
            {"problem":{"ints":[{}],"factors":[{"linear":{"vars":[0],"op":"EQ",
              "wideCoefficients":["1"],"wideBound":"$number"}}]}}
        """.trimIndent())
        val base = "/v1/models/${model.id}"
        val recommendation = client.send("$base/recommend", "{}").objectBody()
        assertEquals("feasible", recommendation.getValue("status").jsonPrimitive.content)
        val values = recommendation.getValue("values").jsonObject
        assertEquals(number, values.getValue("ints.0").jsonPrimitive.content)
        val checked = client.send("$base/evaluate", buildJsonObject { put("values", values) }.toString()).objectBody()
        assertEquals("feasible", checked.getValue("status").jsonPrimitive.content)
        assertEquals(values, checked["values"])
    }

}
