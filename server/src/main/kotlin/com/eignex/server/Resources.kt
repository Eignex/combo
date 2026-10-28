package com.eignex.server

import com.eignex.combo.model.ModelObjective
import com.eignex.combo.decisions.conditionedProblem
import com.eignex.combo.decisions.pinFloats
import com.eignex.combo.model.LinearLearning
import com.eignex.combo.bandit.glm.LinearTrialOptimizer
import com.eignex.combo.bo.TrialOptimizer
import com.eignex.combo.shared.LinearFeatureProjection

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
import com.eignex.combo.decisions.DecisionCoordinates
import com.eignex.combo.decisions.minimizeNative
import com.eignex.combo.decisions.solveNative
import com.eignex.klause.ir.Problem
import com.eignex.klause.solver.objective.LinearObjective
import com.eignex.klause.solver.result.QuadraticOptimizationResult
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

internal class ProblemResource private constructor(
    private val compiled: DecisionCoordinates,
    private val entries: Map<String, SchemaEntry>,
) : ManagedResource {
    constructor(schema: com.eignex.skema.SchemaDef<SchemaEntry>) : this(
        compileInput { DecisionCoordinates.of(schema.compile()) }, schema.entries.also(::requireScalarVariables),
    )
    constructor(problem: Problem) : this(DecisionCoordinates.of(problem), emptyMap())

    override val type = "klause.problem"
    private val native = if (compiled.schema == null) CompiledDecisionSpace.fromProblem(compiled.problem) else null
    private val session = if (native == null) BacktrackSolver(compiled.problem.bake()).session() else null

    fun recommend(goal: ModelObjective?, given: JsonObject, seed: Long, control: OperationControl): JsonObject {
        native?.let { return recommendNative(it, goal, given, seed, control) }
        val pins = realValues(entries, given)
        if (pins.isEmpty()) return recommend(requireNotNull(session), goal, given, seed, control)
        return BacktrackSolver(compiled.pinFloats(pins).bake()).session().use {
            recommend(it, goal, given, seed, control)
        }
    }

    private fun recommend(
        session: com.eignex.klause.solver.Session<BacktrackParams>,
        goal: ModelObjective?, given: JsonObject, seed: Long, control: OperationControl,
    ): JsonObject {
        val params = BacktrackParams(
            maxDecisions = 100_000, randomSeed = seed,
            assumptions = assumptions(compiled, entries, given), cancellation = control.cancellation,
        )
        if (goal == null) return selection(session.solve(params), null)
        val objective = when (val spec = entries.getValue(goal.name)) {
            is IntSpec -> compiled.minimize(IntHandle(goal.name, spec.min.toLong(), spec.max.toLong()))
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
        val result = checkedAssignment(
            compiled, entries, JsonObject(input),
            BacktrackParams(maxDecisions = 100_000, cancellation = control.cancellation),
        )
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

    private fun recommendNative(
        space: CompiledDecisionSpace, goal: ModelObjective?, given: JsonObject, seed: Long, control: OperationControl,
    ): JsonObject {
        if (goal == null) return selection(space.solveNative(readContext(space, given),
            BacktrackParams(maxDecisions = 100_000, randomSeed = seed, cancellation = control.cancellation)), null)
        val objective = goal?.let { space.coordinates.objective(it.name) } ?: LinearObjective()
        val result = space.minimizeNative(
            if (goal?.ascending == false) objective.negated() else objective, readContext(space, given),
            BacktrackParams(maxDecisions = 100_000, randomSeed = seed, cancellation = control.cancellation),
        )
        control.checkpoint()
        return buildJsonObject {
            put("status", when (result) {
                is QuadraticOptimizationResult.Optimal -> if (goal == null) "feasible" else "optimal"
                is QuadraticOptimizationResult.BestFound -> if (goal == null) "feasible" else "best_found"
                is QuadraticOptimizationResult.UnattainedLimit -> "unattained_limit"
                is QuadraticOptimizationResult.Unbounded -> "unbounded"
                is QuadraticOptimizationResult.Infeasible -> "infeasible"
                is QuadraticOptimizationResult.Unknown -> "unknown"
            })
            if (result is QuadraticOptimizationResult.Unknown) put("reason", result.reason.toString())
            if (result is QuadraticOptimizationResult.WithIncumbent) {
                val assignment = values(compiled, entries, result.incumbent.sample)
                put("values", assignment)
                goal?.let { put("objective", buildJsonObject { put("name", it.name); put("value", assignment.getValue(it.name)) }) }
            }
        }
    }

    override fun close() { session?.close() }
}

internal class OptimizerResource private constructor(
    private val space: CompiledDecisionSpace,
    private val entries: Map<String, SchemaEntry>,
    seed: Int,
    maximize: Boolean,
    rewards: SeriesStat<*>? = null,
    learning: LinearLearning? = null,
) : ManagedResource {
    override val type = if (learning == null) "combo.forest_ei" else "combo.linear"
    constructor(
        definition: com.eignex.combo.decisions.DecisionSpaceDef, seed: Int, maximize: Boolean,
        rewards: SeriesStat<*>? = null, learning: LinearLearning? = null,
    ) : this(compileInput { definition.compileSpace() }, definition.toSchemaDef().entries.also(::requireScalarVariables), seed, maximize, rewards, learning)

    constructor(
        problem: Problem, seed: Int, maximize: Boolean,
        rewards: SeriesStat<*>? = null, learning: LinearLearning? = null,
    ) : this(CompiledDecisionSpace.fromProblem(problem), emptyMap(), seed, maximize, rewards, learning)
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
    private val forest = if (learning == null) BayesianOptimizer(
        projection, ForestSurrogate(),
        acquisitionOptimizer = LocalAndRandomSearch(projection, search),
        exploration = search,
        randomSeed = seed,
        outcome = OutcomeSpec(maximize = maximize),
        rewards = rewards,
    ) else null
    private val linearSession = if (space.coordinates.schema != null) BacktrackSolver(space.problem.bake()).session() else null
    private val linear = learning?.let { config ->
        LinearTrialOptimizer(
            LinearFeatureProjection(space), config, search,
            { objective, pinned ->
                requireNotNull(linearSession).minimize(objective, BacktrackParams(
                    maxDecisions = 100_000, assumptions = pinned, cancellation = cancellation,
                )).assignment
            }, seed, maximize, rewards,
            contextOptimizer = { objective, pinned, context ->
                if (space.coordinates.schema == null) {
                    (space.minimizeNative(objective, context, BacktrackParams(
                        maxDecisions = 100_000, assumptions = pinned, cancellation = cancellation,
                    )) as? QuadraticOptimizationResult.WithIncumbent)?.incumbent?.sample
                } else BacktrackSolver(space.conditionedProblem(context).bake()).minimize(
                    objective, BacktrackParams(
                        maxDecisions = 100_000, assumptions = pinned, cancellation = cancellation,
                    ),
                ).assignment
            },
        )
    }
    private val optimizer: TrialOptimizer = linear ?: requireNotNull(forest)

    val observationCount: Long get() = linear?.observationCount ?: requireNotNull(forest).store.revision

    fun recommend(given: JsonObject, target: String, control: OperationControl): JsonObject {
        val context = readContext(space, given)
        requireEvidence()
        activeControl = control
        val sample = try {
            linear?.recommend(context) ?: requireNotNull(forest).recommendOrThrow(context)
        } catch (_: NoFeasibleSampleException) {
            return buildJsonObject { put("status", "unknown") }
        } finally { activeControl = null }
        control.checkpoint()
        return prediction(sample, context, target)
    }

    fun evaluate(input: JsonObject, given: JsonObject, target: String, control: OperationControl): JsonObject {
        val context = readContext(space, given)
        val result = checkedAssignment(
            space.coordinates, entries, JsonObject(input + given),
            BacktrackParams(maxDecisions = 100_000, cancellation = control.cancellation),
        )
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
            put("values", values(space.coordinates, entries, sample.sample))
            put("objective", buildJsonObject { put("name", target); put("estimate", (linear?.predict(sample, context) ?: requireNotNull(forest).predict(sample, context)).let { if (it.isFinite()) JsonPrimitive(it) else JsonNull }) })
        }

    override fun close() {
        linearSession?.close()
    }

    fun propose(given: JsonObject, count: Int, control: OperationControl): com.eignex.combo.bo.TrialBatch {
        val context = readContext(space, given)
        activeControl = control
        return try { optimizer.suggest(count, context) } finally { activeControl = null }
    }

    val trials: List<Trial> get() = optimizer.trials
    fun assignment(trial: Trial): JsonObject = values(space.coordinates, entries, trial.sample.sample)

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
