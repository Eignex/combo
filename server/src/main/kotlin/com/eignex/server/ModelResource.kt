package com.eignex.server

import com.eignex.combo.model.StatisticResource
import com.eignex.combo.model.ModelDefinition
import com.eignex.combo.model.ModelQuery
import com.eignex.combo.model.ModelObservation
import com.eignex.combo.model.ModelObjective
import com.eignex.combo.model.ModelStatistics
import com.eignex.combo.model.StatisticalObservation
import com.eignex.combo.model.InsufficientStatisticalEvidence
import com.eignex.klause.model.VarSpec
import com.eignex.combo.decisions.DecisionCoordinates
import io.ktor.http.HttpStatusCode
import kotlinx.serialization.json.*

internal class ModelResource(private val definition: ModelDefinition) : ManagedResource {
    override val type = "model"
    private val space = definition.space?.also(::requireSupportedSpace)
    private val variables = space?.toSchemaDef()?.entries.orEmpty().filterValues { it is VarSpec }
    private val native = definition.problem?.let(DecisionCoordinates::of)
    private val names = native?.let { it.boolVarIdByName.keys + it.intVarIdByName.keys + it.realHandles.values.map { h -> h.name } } ?: variables.keys
    private val contexts = if (native == null) space?.context.orEmpty().keys else definition.inputs.keys
    private val proposalContexts = mutableMapOf<Long, Set<String>>()
    private val statistics = ModelStatistics(definition)
    private val stat = definition.stat?.let { StatisticResource.create(it, definition.seed) }
    private val problem = if (definition.learned) null else definition.problem?.let(::ProblemResource)
        ?: space?.takeIf { definition.inputs.isEmpty() }?.let { ProblemResource(it.toSchemaDef()) }
    private val optimizer = if (!definition.learned) null else run {
        val rewards = stat?.let { tracker ->
            requireNotNull(tracker.series()) { "an outcome statistic must accept scalar observations" }
        }
        definition.problem?.let {
            OptimizerResource(it, definition.seed, !definition.objective!!.ascending, rewards, definition.learning)
        } ?: OptimizerResource(requireNotNull(space), definition.seed, !definition.objective!!.ascending, rewards, definition.learning)
    }
    val capabilities: List<String> = buildList {
        if (space != null || native != null) { add("evaluate"); add("recommend") }
        if (optimizer != null || stat?.canChoose == true) add("propose")
        if (optimizer != null || stat != null || definition.statistics.isNotEmpty()) add("observe")
        if (optimizer != null) add("cancel")
        if (space == null && native == null && (stat != null || definition.statistics.isNotEmpty())) add("reset")
    }

    fun describe(timestampNanos: Long = System.nanoTime()): JsonObject = buildJsonObject {
        put("definition", wireJson.encodeToJsonElement(definition))
        put("capabilities", JsonArray(capabilities.map(::JsonPrimitive)))
        put("state", snapshot(timestampNanos))
    }

    fun snapshot(timestampNanos: Long = System.nanoTime()): JsonElement {
        if (optimizer != null || definition.statistics.isNotEmpty()) {
            return buildJsonObject {
                optimizer?.let { backend ->
                    put("observations", backend.observationCount)
                    put("proposals", JsonArray(backend.trials.map { proposal(it, backend) }))
                }
                stat?.let { put("stat", it.snapshot(0L)) }
                if (definition.statistics.isNotEmpty()) put("statistics", statistics.snapshot(timestampNanos))
            }
        }
        return stat?.snapshot(timestampNanos) ?: buildJsonObject { }
    }

    fun recommend(query: ModelQuery, control: OperationControl): JsonElement {
        require(query.values.isEmpty() && query.count == 1) { "recommend accepts given and timeoutMillis" }
        if ((space != null || native != null) && optimizer == null) {
            validateGiven(query.given)
            return staticSelection(query.given) { resource, objective, given ->
                resource.recommend(objective, given, definition.seed.toLong(), control)
            }
        }
        optimizer?.let {
            return selection(it.recommend(learnedGiven(query.given), definition.objective!!.name, control), requestContexts(query.given))
        }
        unsupported("recommend")
    }

    fun evaluate(query: ModelQuery, control: OperationControl): JsonElement {
        require(query.count == 1) { "evaluate accepts values, given, and timeoutMillis" }
        validateGiven(query.given)
        require(query.values.keys == names - requestContexts(query.given)) { "values must assign every decision variable" }
        optimizer?.let { return selection(it.evaluate(query.values, learnedGiven(query.given), definition.objective!!.name, control), requestContexts(query.given)) }
        if (space != null || native != null) return staticSelection(query.given) { resource, objective, given ->
            resource.evaluate(query.values + given, objective, control)
        }
        unsupported("evaluate")
    }

    fun propose(request: JsonObject, control: OperationControl): JsonElement {
        optimizer?.let {
            val query = wireJson.decodeFromJsonElement<ModelQuery>(request)
            require(query.values.isEmpty()) { "propose accepts given, count, and timeoutMillis" }
            val proposals = it.propose(learnedGiven(query.given), query.count, control).map { trial ->
                proposalContexts[trial.id] = requestContexts(query.given)
                proposal(trial, it)
            }
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
        if ("statistic" in request) {
            val observation = wireJson.decodeFromJsonElement<StatisticalObservation>(request)
            statistics.observe(observation.statistic, observation.observation)
            return snapshot()
        }
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
        require(space == null && native == null && (stat != null || definition.statistics.isNotEmpty())) { "reset requires standalone statistics or a bandit" }
        statistics.reset()
        stat?.reset()
        return snapshot()
    }

    private fun staticSelection(
        given: JsonObject,
        operation: (ProblemResource, ModelObjective?, JsonObject) -> JsonObject,
    ): JsonObject {
        problem?.let {
            val bound = if (native != null) statisticalContext() else JsonObject(emptyMap())
            val result = operation(it, definition.objective, JsonObject(given + bound))
            return selection(result, requestContexts(given))
        }
        val resolved = try {
            statistics.resolve()
        } catch (error: InsufficientStatisticalEvidence) {
            throw ApiException(HttpStatusCode.Conflict, "insufficient_evidence", error.message.orEmpty())
        }
        val result = ProblemResource(resolved.schema).use { operation(it, resolved.objective, given) }
        return JsonObject(selection(result) + ("inputs" to resolved.inputs))
    }

    private fun validateGiven(given: JsonObject) {
        if (native != null) {
            require(given.keys.all { it in names && it !in definition.inputs }) {
                "given must name native columns and cannot override statistical inputs"
            }
            return
        }
        val supplied = if (definition.learned) contexts - definition.inputs.keys else contexts
        require(given.keys == supplied) { "given must assign every caller context; statistical inputs cannot be overridden" }
    }

    private fun learnedGiven(given: JsonObject): JsonObject {
        validateGiven(given)
        return JsonObject(given + statisticalContext())
    }

    private fun statisticalContext(): JsonObject = try {
            statistics.context()
        } catch (error: InsufficientStatisticalEvidence) {
            throw ApiException(HttpStatusCode.Conflict, "insufficient_evidence", error.message.orEmpty())
        }

    private fun requestContexts(given: JsonObject): Set<String> =
        if (native == null) contexts else given.keys + definition.inputs.keys

    private fun boundInputs(assignment: JsonObject): Map<String, JsonElement> =
        if ((!definition.learned && native == null) || definition.inputs.isEmpty()) emptyMap() else mapOf(
            "inputs" to JsonObject(definition.inputs.map { (name, input) ->
                "$name.${input.value}" to assignment.getValue(name)
            }.toMap()),
        )

    private fun selection(result: JsonObject, givenNames: Set<String> = contexts): JsonObject {
        val assignment = result["values"] as? JsonObject ?: return result
        return JsonObject(result + mapOf(
            "values" to JsonObject(assignment.filterKeys { it in names && it !in givenNames }),
            "given" to JsonObject(assignment.filterKeys { it in givenNames && it !in definition.inputs }),
        ) + boundInputs(assignment) + ((result["objective"] as? JsonObject)?.let {
            mapOf("objective" to JsonObject(it + ("name" to JsonPrimitive(definition.objective!!.name))))
        } ?: emptyMap()))
    }

    private fun proposal(trial: com.eignex.combo.bo.Trial, backend: OptimizerResource): JsonObject = buildJsonObject {
        put("id", trial.id)
        val status = trial.status.name.lowercase()
        put("status", if (status == "completed") "observed" else status)
        val allValues = backend.assignment(trial)
        val givenNames = proposalContexts[trial.id] ?: contexts
        put("values", JsonObject(allValues.filterKeys { it in names && it !in givenNames }))
        put("given", JsonObject(allValues.filterKeys { it in givenNames && it !in definition.inputs }))
        for ((name, value) in boundInputs(allValues)) put(name, value)
        trial.reward?.let { put("observed", buildJsonObject { put(definition.objective!!.name, it) }) }
    }

    private fun unsupported(operation: String): Nothing =
        throw ApiException(HttpStatusCode.BadRequest, "unsupported_operation", "this model does not support $operation")

    override fun close() {
        problem?.close()
        optimizer?.close()
        stat?.close()
        statistics.close()
    }
}
