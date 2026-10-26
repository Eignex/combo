package com.eignex.server

import io.ktor.http.HttpStatusCode
import kotlinx.serialization.json.*

internal class ModelResource(private val definition: ModelDefinition) : ManagedResource {
    override val type = "model"
    private val stat = definition.stat?.let { StatsResource.create(it, definition.seed) }
    private val space = definition.takeIf { it.variables.isNotEmpty() }?.space()
    private val problem = space?.takeUnless { definition.learned }?.let {
        ProblemResource(it.toSchemaDef())
    }
    private val optimizer = space?.takeIf { definition.learned }?.let {
        val rewards = stat?.let { tracker ->
            requireNotNull(tracker.series()) { "an outcome statistic must accept scalar observations" }
        }
        OptimizerResource(it, definition.seed, !definition.objective!!.ascending, rewards)
    }
    val capabilities: List<String> = buildList {
        if (problem != null || optimizer != null) { add("evaluate"); add("recommend") }
        if (optimizer != null || stat?.canChoose == true) add("propose")
        if (optimizer != null || stat != null) add("observe")
        if (optimizer != null) add("cancel")
        if (space == null && stat != null) add("reset")
    }

    fun describe(timestampNanos: Long = System.nanoTime()): JsonObject = buildJsonObject {
        put("definition", wireJson.encodeToJsonElement(definition))
        put("capabilities", JsonArray(capabilities.map(::JsonPrimitive)))
        put("state", snapshot(timestampNanos))
    }

    fun snapshot(timestampNanos: Long = System.nanoTime()): JsonElement {
        optimizer?.let { backend ->
            return buildJsonObject {
                put("observations", backend.observationCount)
                put("proposals", JsonArray(backend.trials.map { proposal(it, backend) }))
                stat?.let { put("stat", it.snapshot(0L)) }
            }
        }
        return stat?.snapshot(timestampNanos) ?: buildJsonObject { }
    }

    fun recommend(query: ModelQuery, control: OperationControl): JsonElement {
        require(query.values.isEmpty() && query.count == 1) { "recommend accepts given and timeoutMillis" }
        problem?.let {
            validateGiven(query.given)
            return selection(it.recommend(definition.objective, query.given, definition.seed.toLong(), control))
        }
        optimizer?.let {
            return selection(it.recommend(query.given, definition.objective!!.name, control))
        }
        unsupported("recommend")
    }

    fun evaluate(query: ModelQuery, control: OperationControl): JsonElement {
        require(query.count == 1) { "evaluate accepts values, given, and timeoutMillis" }
        validateGiven(query.given)
        val decisions = definition.variables.filterValues { it.role == "decision" }.keys
        require(query.values.keys == decisions) { "values must assign every decision variable" }
        optimizer?.let { return selection(it.evaluate(query.values, query.given, definition.objective!!.name, control)) }
        problem?.let { return selection(it.evaluate(query.values + query.given, definition.objective, control)) }
        unsupported("evaluate")
    }

    fun propose(request: JsonObject, control: OperationControl): JsonElement {
        optimizer?.let {
            val query = wireJson.decodeFromJsonElement<ModelQuery>(request)
            require(query.values.isEmpty()) { "propose accepts given, count, and timeoutMillis" }
            val proposals = it.propose(query.given, query.count, control).map { trial -> proposal(trial, it) }
            return buildJsonObject {
                put("status", if (proposals.size == query.count) "ready" else if (proposals.isEmpty()) "unavailable" else "partial")
                put("proposals", JsonArray(proposals))
            }
        }
        if (stat?.canChoose == true) return buildJsonObject {
            put("status", "ready")
            put("proposals", buildJsonArray { add(buildJsonObject { put("values", stat.choose(request)) }) })
        }
        unsupported("propose")
    }

    fun observe(request: JsonObject): JsonElement {
        optimizer?.let {
            val observation = wireJson.decodeFromJsonElement<ModelObservation>(request)
            val target = definition.objective!!.name
            require(observation.values.keys == setOf(target)) { "values must contain the observed outcome '$target'" }
            val value = observation.values.getValue(target).jsonPrimitive
            val number = requireNotNull(value.takeUnless { it.isString }?.doubleOrNull?.takeIf { it.isFinite() }) {
                "'$target' must be a finite number"
            }
            it.complete(observation.proposal, number, observation.weight, observation.noiseVariance)
            return snapshot()
        }
        stat?.let { return it.update(request) }
        unsupported("observe")
    }

    fun cancel(proposal: Long): JsonElement {
        optimizer?.let { it.cancel(proposal); return snapshot() }
        unsupported("cancel")
    }

    fun reset(): JsonElement {
        require(space == null && stat != null) { "reset requires a standalone statistic or bandit" }
        return stat.reset()
    }

    private fun validateGiven(given: JsonObject) {
        val contexts = definition.variables.filterValues { it.role == "context" }.keys
        require(given.keys == contexts) { "given must assign every context variable" }
    }

    private fun selection(result: JsonObject): JsonObject {
        val assignment = result["values"] as? JsonObject ?: return result
        return JsonObject(result + mapOf(
            "values" to JsonObject(assignment.filterKeys { definition.variables[it]?.role == "decision" }),
            "given" to JsonObject(assignment.filterKeys { definition.variables[it]?.role == "context" }),
        ))
    }

    private fun proposal(trial: com.eignex.combo.bo.Trial, backend: OptimizerResource): JsonObject = buildJsonObject {
        put("id", trial.id)
        val status = trial.status.name.lowercase()
        put("status", if (status == "completed") "observed" else status)
        val allValues = backend.assignment(trial)
        put("values", JsonObject(allValues.filterKeys { definition.variables[it]?.role == "decision" }))
        put("given", JsonObject(allValues.filterKeys { definition.variables[it]?.role == "context" }))
        trial.reward?.let { put("observed", buildJsonObject { put(definition.objective!!.name, it) }) }
    }

    private fun unsupported(operation: String): Nothing =
        throw ApiException(HttpStatusCode.BadRequest, "unsupported_operation", "this model does not support $operation")

    override fun close() {
        problem?.close()
        optimizer?.close()
        stat?.close()
    }
}
