package com.eignex.server

import com.eignex.kumulant.stat.decay.EwmaMeanStat
import io.ktor.client.request.delete
import io.ktor.client.request.get
import io.ktor.client.request.post
import io.ktor.client.statement.bodyAsText
import io.ktor.http.HttpStatusCode
import io.ktor.server.testing.testApplication
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.serialization.json.*
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class StatsRoutesTest {
    @Test
    fun `canonical transformed ewma retains observations across requests`() = testApplication {
        application { resourceApi() }
        val stat = client.create("stats", """
            {"transform":{"expr":{"add":["x",1.0]},"inner":{"ewma":{"alpha":0.1}}}}
        """.trimIndent())
        val first = client.send("/v1/stats/${stat.id}/update", """{"value":42.0}""")
        assertEquals(HttpStatusCode.OK, first.status, first.bodyAsText())
        assertEquals(43.0, first.objectBody().getValue("mean").jsonPrimitive.double)
        assertEquals(1.0, first.objectBody().getValue("totalWeights").jsonPrimitive.double)

        val second = client.send("/v1/stats/${stat.id}/update", """{"value":84.0}""")
        val expected = EwmaMeanStat(0.1).apply { update(43.0); update(85.0) }.read()
        assertEquals(expected.mean, second.objectBody().getValue("mean").jsonPrimitive.double, 1e-10)
        repeat(2) { assertEquals(second.objectBody(), client.get("/v1/stats/${stat.id}").objectBody()) }
        val another = client.create("stats", """{"ewma":{"alpha":0.1}}""")
        assertEquals(0.0, client.get("/v1/stats/${another.id}").objectBody().getValue("totalWeights").jsonPrimitive.double)
    }

    @Test
    fun `native weights allow zero weight and additive downdates`() = testApplication {
        application { resourceApi() }
        val stat = client.create("stats", """{"sum":{}}""")
        for ((observation, expected) in listOf(
            """{"value":3,"weight":2}""" to 6.0,
            """{"value":100,"weight":0}""" to 6.0,
            """{"value":3,"weight":-1}""" to 3.0,
        )) {
            val result = client.send("/v1/stats/${stat.id}/update", observation)
            assertEquals(HttpStatusCode.OK, result.status, result.bodyAsText())
            assertEquals(expected, result.objectBody().getValue("sum").jsonPrimitive.double)
        }
    }

    @Test
    fun `grouped stats return nested native snapshots`() = testApplication {
        application { resourceApi() }
        val stat = client.create("stats", """
            {"group":{"stats":{"average":{"mean":{}},"nested":{"group":{"stats":{"sum":{"sum":{}}}}}}}}
        """.trimIndent())
        val response = client.send("/v1/stats/${stat.id}/update", """{"value":7}""")
        assertEquals(HttpStatusCode.OK, response.status, response.bodyAsText())
        val results = response.objectBody().getValue("results").jsonObject
        assertEquals(7.0, results.getValue("average").jsonObject.getValue("mean").jsonPrimitive.double)
        val nested = results.getValue("nested").jsonObject.getValue("results").jsonObject
        assertEquals(7.0, nested.getValue("sum").jsonObject.getValue("sum").jsonPrimitive.double)
    }

    @Test
    fun `discrete paired vector and regression observations use the same lifecycle`() = testApplication {
        application { resourceApi() }
        val cases = listOf(
            """{"asDiscrete":{"inner":{"sum":{}}}}""" to """{"value":3}""",
            """{"covariance":{}}""" to """{"x":2,"y":3}""",
            """{"vectorize":{"dimensions":2,"template":{"sum":{}}}}""" to """{"x":[2,3]}""",
            """{"bayesianRegression":{"featureSize":2}}""" to """{"x":[1,0],"y":3}""",
        )
        for ((spec, observation) in cases) {
            val stat = client.create("stats", spec)
            val before = client.get("/v1/stats/${stat.id}").objectBody()
            val response = client.send("/v1/stats/${stat.id}/update", observation)
            assertEquals(HttpStatusCode.OK, response.status, response.bodyAsText())
            assertTrue(before != response.objectBody(), spec)
            assertEquals(response.objectBody(), client.get("/v1/stats/${stat.id}").objectBody())
        }
    }

    @Test
    fun `explicit timestamps support time dependent statistics`() = testApplication {
        application { resourceApi() }
        val stat = client.create("stats", """{"decayingMean":{"halfLifeMillis":1000}}""")
        client.send("/v1/stats/${stat.id}/update", """{"value":1,"timestampNanos":1000000000}""")
        val response = client.send("/v1/stats/${stat.id}/update", """{"value":3,"timestampNanos":2000000000}""")
        assertEquals(HttpStatusCode.OK, response.status, response.bodyAsText())
        assertEquals(7.0 / 3.0, response.objectBody().getValue("mean").jsonPrimitive.double, 1e-10)
        val read = client.get("/v1/stats/${stat.id}?timestampNanos=2000000000")
        assertEquals(response.objectBody(), read.objectBody())
    }

    @Test
    fun `ordinary bandits choose using retained rewards`() = testApplication {
        application { resourceApi() }
        val stat = client.create("stats", """{"multiArmed":{"nbrArms":2,"policy":{"greedy":{}}}}""")
        for ((arm, reward) in listOf(0 to 1.0, 1 to 4.0)) {
            val update = client.send("/v1/stats/${stat.id}/update", """{"arm":$arm,"reward":$reward}""")
            assertEquals(HttpStatusCode.OK, update.status, update.bodyAsText())
        }
        repeat(2) {
            val choice = client.send("/v1/stats/${stat.id}/choose", "{}")
            assertEquals(HttpStatusCode.OK, choice.status, choice.bodyAsText())
            assertEquals(1, choice.objectBody().getValue("arm").jsonPrimitive.int)
        }
        assertEquals(2, client.get("/v1/stats/${stat.id}").objectBody().getValue("arms").jsonArray.size)
    }

    @Test
    fun `univariate policy families accept native specs without mode wrappers`() = testApplication {
        application { resourceApi() }
        val specs = listOf(
            """{"multiArmed":{"nbrArms":2,"policy":{"thompsonSampling":{"arm":{"bernoulliArm":{}},"posterior":{"betaPosterior":{}}}}}}""",
            """{"multiArmed":{"nbrArms":2,"policy":{"ucb1":{}}}}""",
            """{"rouletteWheel":{"nbrArms":2}}""",
            """{"boltzmann":{"nbrArms":2}}""",
            """{"exp3":{"nbrArms":2}}""",
            """{"topTwoThompson":{"nbrArms":2,"policy":{"arm":{"bernoulliArm":{}},"posterior":{"betaPosterior":{}}}}}""",
        )
        for (spec in specs) {
            val stat = client.create("stats?seed=42", spec)
            val choice = client.send("/v1/stats/${stat.id}/choose", "{}")
            assertEquals(HttpStatusCode.OK, choice.status, choice.bodyAsText())
            val arm = choice.objectBody().getValue("arm").jsonPrimitive.int
            val update = client.send("/v1/stats/${stat.id}/update", """{"arm":$arm,"reward":1}""")
            assertEquals(HttpStatusCode.OK, update.status, update.bodyAsText())
        }
    }

    @Test
    fun `contextual bandits learn different choices for different contexts`() = testApplication {
        application { resourceApi() }
        val stat = client.create("stats", """
            {"regressionContextual":{"nbrArms":2,"regression":{"bayesianRegression":{"featureSize":2}},
            "posterior":{"linUcb":{}},"exploration":0.0}}
        """.trimIndent())
        for ((arm, x) in listOf(0 to "[1,0]", 1 to "[0,1]")) {
            val response = client.send("/v1/stats/${stat.id}/update", """{"arm":$arm,"x":$x,"reward":4}""")
            assertEquals(HttpStatusCode.OK, response.status, response.bodyAsText())
        }
        for ((arm, x) in listOf(0 to "[1,0]", 1 to "[0,1]")) {
            val response = client.send("/v1/stats/${stat.id}/choose", """{"x":$x}""")
            assertEquals(HttpStatusCode.OK, response.status, response.bodyAsText())
            assertEquals(arm, response.objectBody().getValue("arm").jsonPrimitive.int)
        }
        val before = client.get("/v1/stats/${stat.id}").objectBody()
        for (body in listOf("{}", """{"x":[1]}""", """{"x":[1,0,0]}""")) {
            assertEquals(HttpStatusCode.BadRequest, client.send("/v1/stats/${stat.id}/choose", body).status)
        }
        val badUpdate = client.send("/v1/stats/${stat.id}/update", """{"arm":0,"x":[1],"reward":10}""")
        assertEquals(HttpStatusCode.BadRequest, badUpdate.status)
        assertEquals(before, client.get("/v1/stats/${stat.id}").objectBody())
    }

    @Test
    fun `tracked bandits expose their native observation statistics`() = testApplication {
        application { resourceApi() }
        val stat = client.create("stats", """
            {"tracked":{"inner":{"multiArmed":{"nbrArms":2,"policy":{"greedy":{}}}},
            "chooseTemplate":{"count":{}},"updateArmRewardTemplate":{"covariance":{}}}}
        """.trimIndent())
        val choice = client.send("/v1/stats/${stat.id}/choose", "{}")
        val arm = choice.objectBody().getValue("arm").jsonPrimitive.int
        val response = client.send("/v1/stats/${stat.id}/update", """{"arm":$arm,"reward":1}""")
        assertEquals(HttpStatusCode.OK, response.status, response.bodyAsText())
        val snapshot = response.objectBody()
        assertEquals(1.0, snapshot.getValue("choose").jsonObject.getValue("sum").jsonPrimitive.double)
        assertTrue("updateArmReward" in snapshot)
        assertEquals(2, snapshot.getValue("inner").jsonObject.getValue("arms").jsonArray.size)
    }

    @Test
    fun `tracked contextual bandits retain global and observation models`() = testApplication {
        application { resourceApi() }
        val stat = client.create("stats", """
            {"tracked":{
              "inner":{"regressionContextual":{
                "nbrArms":2,"regression":{"bayesianRegression":{"featureSize":2}},
                "globalRegression":{"bayesianRegression":{"featureSize":2}},
                "posterior":{"linUcb":{}},"exploration":0.0}},
              "contextFeatureSize":2,
              "chooseTemplate":{"bayesianRegression":{"featureSize":2}},
              "updateJointTemplate":{"bayesianRegression":{"featureSize":3}},
              "updateMarginalTemplate":{"bayesianRegression":{"featureSize":2}},
              "updateArmRewardTemplate":{"covariance":{}}
            }}
        """.trimIndent())
        val choice = client.send("/v1/stats/${stat.id}/choose", """{"x":[1,0]}""")
        assertEquals(HttpStatusCode.OK, choice.status, choice.bodyAsText())
        val arm = choice.objectBody().getValue("arm").jsonPrimitive.int
        val response = client.send("/v1/stats/${stat.id}/update", """{"arm":$arm,"x":[1,0],"reward":2}""")
        assertEquals(HttpStatusCode.OK, response.status, response.bodyAsText())
        val snapshot = response.objectBody()
        assertTrue("global" in snapshot.getValue("inner").jsonObject)
        for (name in listOf("choose", "updateJoint", "updateMarginal", "updateArmReward")) {
            assertEquals(1.0, snapshot.getValue(name).jsonObject.getValue("totalWeights").jsonPrimitive.double)
        }
    }

    @Test
    fun `invalid native specs are rejected without retaining resources`() = testApplication {
        application { resourceApi() }
        val specs = listOf(
            "{}", "[]", """{"ewma":{"alpha":-0.1}}""", """{"ewma":{"alpha":0.1,"typo":1}}""",
            """{"transform":{"expr":{"add":["x",1,2]},"inner":{"sum":{}}}}""",
            """{"group":{"stats":{"bad":{"covariance":{}}}}}""",
            """{"regressionContextual":{"nbrArms":2,"regression":{"bayesianRegression":{"featureSize":1}},"posterior":{"pointPosterior":{}}}}""",
        )
        for (spec in specs) {
            val response = client.send("/v1/stats", spec)
            assertEquals(HttpStatusCode.BadRequest, response.status, spec)
        }
        assertEquals(HttpStatusCode.BadRequest, client.send("/v1/stats?seed=invalid", """{"mean":{}}""").status)
        assertEquals("[]", client.get("/v1/stats").bodyAsText())
    }

    @Test
    fun `invalid observations do not change existing evidence`() = testApplication {
        application { resourceApi() }
        val stat = client.create("stats", """{"sum":{}}""")
        client.send("/v1/stats/${stat.id}/update", """{"value":3}""")
        val before = client.get("/v1/stats/${stat.id}").objectBody()
        for (body in listOf(
            "{}", "[]", """{"value":"4"}""", """{"value":null}""", """{"value":1e999}""",
            """{"value":4,"weight":"2"}""", """{"value":4,"timestampNanos":1.5}""",
            """{"value":4,"arm":0}""", """{"value":4,"typo":1}""",
        )) {
            val response = client.send("/v1/stats/${stat.id}/update", body)
            assertEquals(HttpStatusCode.BadRequest, response.status, body)
        }
        assertEquals(HttpStatusCode.BadRequest, client.send("/v1/stats/${stat.id}/choose", "{}").status)
        assertEquals(before, client.get("/v1/stats/${stat.id}").objectBody())
    }

    @Test
    fun `undefined statistical results return standard JSON nulls`() = testApplication {
        application { resourceApi() }
        val stat = client.create("stats", """{"transform":{"expr":{"div":["x",0]},"inner":{"mean":{}}}}""")
        val response = client.send("/v1/stats/${stat.id}/update", """{"value":1}""")
        assertEquals(HttpStatusCode.OK, response.status, response.bodyAsText())
        assertEquals(JsonNull, response.objectBody().getValue("mean"))
        assertEquals(1.0, response.objectBody().getValue("totalWeights").jsonPrimitive.double)
    }

    @Test
    fun `concurrent updates preserve evidence and reset only affects one resource`() = testApplication {
        application { resourceApi() }
        val stat = client.create("stats", """{"sum":{}}""")
        val other = client.create("stats", """{"sum":{}}""")
        val statuses = coroutineScope {
            (1..20).map { async { client.send("/v1/stats/${stat.id}/update", """{"value":1}""").status } }.awaitAll()
        }
        assertTrue(statuses.all { it == HttpStatusCode.OK })
        assertEquals(20.0, client.get("/v1/stats/${stat.id}").objectBody().getValue("sum").jsonPrimitive.double)
        assertEquals(0.0, client.get("/v1/stats/${other.id}").objectBody().getValue("sum").jsonPrimitive.double)
        assertEquals(HttpStatusCode.OK, client.post("/v1/stats/${stat.id}/reset").status)
        assertEquals(0.0, client.get("/v1/stats/${stat.id}").objectBody().getValue("sum").jsonPrimitive.double)
        assertEquals(HttpStatusCode.NoContent, client.delete("/v1/stats/${stat.id}").status)
        assertEquals(HttpStatusCode.NotFound, client.get("/v1/stats/${stat.id}").status)
    }
}
