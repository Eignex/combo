package com.eignex.server

import com.eignex.combo.bo.RandomForestOptimizer
import com.eignex.combo.bo.SearchTransform
import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.compileSpace
import com.eignex.combo.decisions.context
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.backtrack.BacktrackSolver
import com.eignex.klause.compile.CompiledSchema
import com.eignex.klause.compile.compile
import com.eignex.klause.localsearch.LocalSearchParams
import com.eignex.klause.model.BoolSpec
import com.eignex.klause.model.FloatSpec
import com.eignex.klause.model.IntSpec
import com.eignex.klause.model.NominalSpec
import com.eignex.klause.model.PresenceSpec
import com.eignex.klause.model.SchemaEntry
import com.eignex.klause.model.VarSpec
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.propagation.bake
import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.FloatHandle
import com.eignex.klause.schema.IntHandle
import com.eignex.klause.schema.NominalHandle
import com.eignex.klause.solver.Sample
import com.eignex.klause.solver.SolveResult
import com.eignex.kumulant.stat.summary.SummaryStat
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import kotlinx.serialization.json.put

internal fun executeJob(request: JobRequest, control: JobControl): JsonObject = when (request) {
    is KlauseSolve -> solve(request, control)
    is KumulantSummary -> summarize(request, control)
    is ComboSuggest -> suggest(request, control)
}

private fun solve(request: KlauseSolve, control: JobControl): JsonObject {
    requireScalarVariables(request.schema.entries)
    val compiled = compileInput { request.schema.compile() }
    control.checkpoint()
    val result = BacktrackSolver(compiled.problem.bake()).solve(
        BacktrackParams(maxDecisions = request.maxDecisions, randomSeed = request.seed, cancellation = control.cancellation),
    )
    control.checkpoint()
    return buildJsonObject {
        when (result) {
            is SolveResult.Sat -> {
                put("verdict", "sat")
                put("values", values(compiled, request.schema.entries, result.assignment))
            }
            is SolveResult.Unsat -> put("verdict", "unsat")
            is SolveResult.Unknown -> {
                put("verdict", "unknown")
                put("reason", result.reason.toString())
            }
        }
    }
}

private fun summarize(request: KumulantSummary, control: JobControl): JsonObject {
    val stat = SummaryStat()
    for (value in request.values) {
        control.checkpoint()
        stat.update(value)
    }
    val result = stat.read()
    require(listOf(result.mean, result.variance, result.min, result.max).all(Double::isFinite)) {
        "observations overflowed the summary"
    }
    return buildJsonObject {
        put("count", request.values.size)
        put("mean", result.mean)
        put("variance", result.variance)
        put("min", result.min)
        put("max", result.max)
    }
}

private fun suggest(request: ComboSuggest, control: JobControl): JsonObject {
    val entries = request.space.toSchemaDef().entries
    requireScalarVariables(entries)
    val space = compileInput { request.space.compileSpace() }
    val search = HybridDescentSession.klause(
        space,
        localSearch = LocalSearchParams(maxFlips = 2000, cancellation = control.cancellation),
        backtrack = BacktrackParams(maxDecisions = 10_000, cancellation = control.cancellation),
        candidateBatchSize = 64,
        poolCapacity = 256,
    )
    val optimizer = RandomForestOptimizer(
        SearchTransform(space), search = search, randomSeed = request.seed, maximize = request.maximize,
    )
    for (observation in request.observations) {
        control.checkpoint()
        val context = readContext(space, observation.context)
        val sample = observedSample(space, entries, observation.values, context, control)
        optimizer.train(BanditSample(sample), context, observation.reward, observation.weight)
    }
    control.checkpoint()
    val batch = optimizer.suggest(request.count, readContext(space, request.context))
    control.checkpoint()
    return buildJsonObject {
        put("termination", batch.termination.name)
        put("suggestions", buildJsonArray {
            for (trial in batch) add(values(space.compiled, entries, trial.sample.sample))
        })
    }
}

private fun requireScalarVariables(entries: Map<String, SchemaEntry>) {
    for ((name, entry) in entries) {
        require(entry !is VarSpec || entry == BoolSpec || entry is IntSpec || entry is FloatSpec ||
            entry is NominalSpec || entry is PresenceSpec) { "unsupported variable type at '$name'" }
    }
}

private inline fun <T> compileInput(compile: () -> T): T = try {
    compile()
} catch (error: IllegalStateException) {
    throw IllegalArgumentException(error.message ?: "invalid model", error)
}

private fun values(compiled: CompiledSchema, entries: Map<String, SchemaEntry>, sample: Sample): JsonObject =
    buildJsonObject {
        for ((name, spec) in entries) {
            when (spec) {
                BoolSpec, is PresenceSpec -> put(name, compiled.decode(BoolHandle(name), sample))
                is IntSpec -> put(name, compiled.decode(IntHandle(name, spec.min, spec.max), sample))
                is NominalSpec -> put(name, compiled.decode(NominalHandle(name, spec.labels), sample))
                is FloatSpec -> put(name, compiled.decode(FloatHandle(name, spec.min, spec.max), sample))
                else -> Unit
            }
        }
    }

private fun readContext(space: CompiledDecisionSpace, input: JsonObject): Context {
    val names = (space.contextBools.map { it.name } + space.contextInts.map { it.name }).toSet()
    require(input.keys.all { it in names }) { "unknown context field" }
    return context {
        for (handle in space.contextBools) {
            val value = input[handle.name]
            require(value != null || handle.isOptional) { "missing context '${handle.name}'" }
            if (value != null) {
                require(!value.jsonPrimitive.isString) { "context '${handle.name}' must be Boolean" }
                set(handle, value.jsonPrimitive.boolean)
            }
        }
        for (handle in space.contextInts) {
            val value = input[handle.name]
            require(value != null || handle.isOptional) { "missing context '${handle.name}'" }
            if (value != null) {
                require(!value.jsonPrimitive.isString) { "context '${handle.name}' must be an integer" }
                val number = value.jsonPrimitive.int
                require(number in handle.min..handle.max) { "context '${handle.name}' outside its bounds" }
                set(handle, number)
            }
        }
    }
}

private fun observedSample(
    space: CompiledDecisionSpace,
    entries: Map<String, SchemaEntry>,
    input: JsonObject,
    context: Context,
    control: JobControl,
): Sample {
    require(entries.values.none { it is FloatSpec }) { "observation import currently supports finite spaces only" }
    val variables = entries.filterValues { it is VarSpec }
    require(input.keys == variables.keys) { "observation values must contain every compiled variable exactly once" }
    val compiled = space.compiled
    var assumptions = Assumptions.None
    for ((name, spec) in variables) {
        val value = input.getValue(name).jsonPrimitive
        assumptions = when (spec) {
            BoolSpec, is PresenceSpec -> {
                require(!value.isString) { "observation '$name' must be Boolean" }
                assumptions.withBool(compiled.boolVarIdByName.getValue(name), value.boolean)
            }
            is IntSpec -> {
                require(!value.isString) { "observation '$name' must be an integer" }
                val number = value.long
                require(number in spec.min.toLong()..spec.max.toLong()) { "observation '$name' outside its bounds" }
                assumptions.withInt(compiled.intVarIdByName.getValue(name), number)
            }
            is NominalSpec -> {
                require(value.isString && value.content in spec.labels) { "invalid label for '$name'" }
                var pinned = assumptions
                for ((label, id) in compiled.nominalIndicators.getValue(name)) pinned = pinned.withBool(id, label == value.content)
                pinned
            }
            else -> throw IllegalArgumentException("unsupported observation variable '$name'")
        }
    }
    require(matchesContext(assumptions, space.assumptionsFor(context))) { "observation does not match its context" }
    val result = BacktrackSolver(compiled.problem.bake()).solve(
        BacktrackParams(maxDecisions = 100_000, assumptions = assumptions, cancellation = control.cancellation),
    )
    control.checkpoint()
    require(result is SolveResult.Sat) { "observation is infeasible or could not be validated within the search budget" }
    return result.assignment
}

private fun matchesContext(values: Assumptions, expected: Assumptions): Boolean =
    expected.bools.all { (id, value) -> values.boolValueOrNull(id) == value } &&
        expected.ints.all { (id, value) -> values.intValueOrNull(id) == value }
