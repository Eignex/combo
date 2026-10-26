package com.eignex.server

import com.eignex.combo.bo.BayesianOptimizer
import com.eignex.combo.bo.ForestSurrogate
import com.eignex.combo.bo.LocalAndRandomSearch
import com.eignex.combo.bo.OutcomeSpec
import com.eignex.combo.bo.SearchTransform
import com.eignex.combo.bo.Trial
import com.eignex.combo.bo.TrialStatus
import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.compileSpace
import com.eignex.combo.shared.NoFeasibleSampleException
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.backtrack.BacktrackSolver
import com.eignex.klause.compile.compile
import com.eignex.klause.localsearch.LocalSearchParams
import com.eignex.klause.propagation.bake
import com.eignex.klause.solver.SolveResult
import com.eignex.klause.solver.result.MinimizeResult
import com.eignex.klause.model.*
import com.eignex.klause.schema.*
import com.eignex.kumulant.core.SeriesStat
import kotlinx.serialization.json.*
import com.eignex.klause.util.Cancellation
import io.ktor.http.HttpStatusCode
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

internal class ProblemResource(schema: com.eignex.skema.SchemaDef<SchemaEntry>) : ManagedResource {
    override val type = "klause.problem"
    private val entries = schema.entries.also(::requireScalarVariables)
    private val compiled = compileInput { schema.compile() }
    private val session = BacktrackSolver(compiled.problem.bake()).session()

    fun recommend(goal: ModelObjective?, given: JsonObject, seed: Long, control: OperationControl): JsonObject {
        val params = BacktrackParams(
            maxDecisions = 100_000, randomSeed = seed,
            assumptions = assumptions(compiled, entries, given), cancellation = control.cancellation,
        )
        if (goal == null) return selection(session.solve(params), null)
        val objective = when (val spec = entries.getValue(goal.name)) {
            is IntSpec -> compiled.minimize(IntHandle(goal.name, spec.min, spec.max))
            BoolSpec -> compiled.minimize(BoolHandle(goal.name))
            is FloatSpec -> compiled.minimize(FloatHandle(goal.name, spec.min, spec.max))
            else -> throw IllegalArgumentException("the objective must be numeric or Boolean")
        }
        val result = session.minimize(if (goal.ascending) objective else objective.negated(), params)
        control.checkpoint()
        return buildJsonObject {
            put("status", when (result) {
                is MinimizeResult.Optimal -> if (entries.values.any { it is FloatSpec }) "best_found" else "optimal"
                is MinimizeResult.BestFound -> "best_found"
                is MinimizeResult.Infeasible -> "infeasible"
                is MinimizeResult.Unbounded -> "unbounded"
                is MinimizeResult.Unknown -> "unknown"
            })
            result.assignment?.let { sample ->
                val assignment = values(compiled, entries, sample)
                put("values", assignment)
                put("objective", buildJsonObject { put("name", goal.name); put("value", assignment.getValue(goal.name)) })
            }
        }
    }

    fun evaluate(input: Map<String, JsonElement>, goal: ModelObjective?, control: OperationControl): JsonObject {
        val result = checkedAssignment(compiled, entries, JsonObject(input)) { pinned ->
            session.solve(BacktrackParams(maxDecisions = 100_000, assumptions = pinned, cancellation = control.cancellation))
        }
        control.checkpoint()
        return selection(result, goal)
    }

    private fun selection(result: SolveResult, goal: ModelObjective?): JsonObject = buildJsonObject {
        put("status", when (result) {
            is SolveResult.Sat -> "feasible"
            is SolveResult.Unsat -> "infeasible"
            is SolveResult.Unknown -> "unknown"
        })
        if (result is SolveResult.Sat) {
            val assignment = values(compiled, entries, result.assignment)
            put("values", assignment)
            goal?.let { put("objective", buildJsonObject { put("name", it.name); put("value", assignment.getValue(it.name)) }) }
        }
    }

    override fun close() = session.close()
}

internal class OptimizerResource(
    definition: com.eignex.combo.decisions.DecisionSpaceDef,
    seed: Int,
    maximize: Boolean,
    rewards: SeriesStat<*>? = null,
) : ManagedResource {
    override val type = "combo.forest_ei"
    private val entries = definition.toSchemaDef().entries.also(::requireScalarVariables)
    private val space: CompiledDecisionSpace = compileInput { definition.compileSpace() }
    private var activeControl: OperationControl? = null
    private val cancellation = Cancellation { activeControl?.cancellation?.invoke() ?: false }
    private val search = HybridDescentSession.klause(
        space,
        localSearch = LocalSearchParams(maxFlips = 2000, cancellation = cancellation),
        backtrack = BacktrackParams(maxDecisions = 10_000, cancellation = cancellation),
        candidateBatchSize = 64,
        poolCapacity = 256,
    )
    private val projection = SearchTransform(space)
    private val optimizer = BayesianOptimizer(
        projection, ForestSurrogate(),
        acquisitionOptimizer = LocalAndRandomSearch(projection, search),
        exploration = search,
        randomSeed = seed,
        outcome = OutcomeSpec(maximize = maximize),
        rewards = rewards,
    )

    val observationCount: Long get() = optimizer.store.revision
    private val validationSession = BacktrackSolver(space.compiled.problem.bake()).session()

    fun recommend(given: JsonObject, target: String, control: OperationControl): JsonObject {
        val context = readContext(space, given)
        requireEvidence()
        activeControl = control
        val sample = try {
            optimizer.recommendOrThrow(context)
        } catch (_: NoFeasibleSampleException) {
            return buildJsonObject { put("status", "unknown") }
        } finally { activeControl = null }
        control.checkpoint()
        return prediction(sample, context, target)
    }

    fun evaluate(input: JsonObject, given: JsonObject, target: String, control: OperationControl): JsonObject {
        val context = readContext(space, given)
        val result = checkedAssignment(space.compiled, entries, JsonObject(input + given)) { pinned ->
            validationSession.solve(BacktrackParams(maxDecisions = 100_000, assumptions = pinned, cancellation = control.cancellation))
        }
        control.checkpoint()
        return when (result) {
            is SolveResult.Sat -> {
                requireEvidence()
                prediction(BanditSample(result.assignment), context, target)
            }
            is SolveResult.Unsat -> buildJsonObject { put("status", "infeasible") }
            is SolveResult.Unknown -> buildJsonObject { put("status", "unknown") }
        }
    }

    private fun requireEvidence() {
        if (observationCount == 0L) throw ApiException(
            HttpStatusCode.Conflict, "insufficient_evidence", "observe an outcome before requesting an estimate",
        )
    }

    private fun prediction(sample: BanditSample, context: com.eignex.combo.decisions.Context, target: String): JsonObject =
        buildJsonObject {
            put("status", "estimated")
            put("values", values(space.compiled, entries, sample.sample))
            put("objective", buildJsonObject { put("name", target); put("estimate", optimizer.predict(sample, context).let { if (it.isFinite()) JsonPrimitive(it) else JsonNull }) })
        }

    override fun close() = validationSession.close()

    fun propose(given: JsonObject, count: Int, control: OperationControl): com.eignex.combo.bo.TrialBatch {
        val context = readContext(space, given)
        activeControl = control
        return try { optimizer.suggest(count, context) } finally { activeControl = null }
    }

    val trials: List<Trial> get() = optimizer.trials
    fun assignment(trial: Trial): JsonObject = values(space.compiled, entries, trial.sample.sample)

    fun complete(id: Long, value: Double, weight: Double, noiseVariance: Double?) {
        pending(id)
        optimizer.complete(id, value, weight, noiseVariance)
    }

    fun cancel(id: Long) {
        pending(id)
        optimizer.cancel(id)
    }

    private fun trial(id: Long): Trial = optimizer.trials.firstOrNull { it.id == id }
        ?: throw ApiException(HttpStatusCode.NotFound, "unknown_proposal", "unknown proposal")

    private fun pending(id: Long) {
        if (trial(id).status != TrialStatus.Pending) {
            throw ApiException(HttpStatusCode.Conflict, "proposal_closed", "proposal is already closed")
        }
    }

}

internal class ServerState(private val runner: OperationRunner = OperationRunner(), capacity: Int = 1000) : AutoCloseable {
    val models = ResourceStore<ModelResource>(runner, capacity)

    override fun close() {
        models.close()
        runner.close()
    }
}
