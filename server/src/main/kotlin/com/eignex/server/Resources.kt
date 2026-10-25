package com.eignex.server

import com.eignex.combo.bo.BayesianOptimizer
import com.eignex.combo.bo.ForestSurrogate
import com.eignex.combo.bo.LocalAndRandomSearch
import com.eignex.combo.bo.OutcomeSpec
import com.eignex.combo.bo.SearchTransform
import com.eignex.combo.bo.Trial
import com.eignex.combo.bo.TrialStatus
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.compileSpace
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.backtrack.BacktrackSolver
import com.eignex.klause.compile.compile
import com.eignex.klause.localsearch.LocalSearchParams
import com.eignex.klause.propagation.bake
import com.eignex.klause.solver.SolveResult
import com.eignex.klause.solver.result.SampleResult
import com.eignex.klause.util.Cancellation
import com.eignex.kumulant.stat.summary.SummaryResult
import com.eignex.kumulant.stat.summary.SummaryStat
import io.ktor.http.HttpStatusCode
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.encodeToJsonElement
import kotlinx.serialization.json.put

internal class ProblemResource(private val definition: CreateProblem) : ManagedResource {
    override val type = "klause.problem"
    private val entries = definition.schema.entries.also(::requireScalarVariables)
    private val compiled = compileInput { definition.schema.compile() }
    private val session = BacktrackSolver(compiled.problem.bake()).session()

    fun describe(id: String): JsonObject = buildJsonObject {
        put("id", id)
        put("type", type)
        put("schema", wireJson.encodeToJsonElement(definition.schema))
    }

    fun solve(request: ProblemQuery, control: OperationControl): JsonObject {
        val result = session.solve(params(request, control))
        control.checkpoint()
        return buildJsonObject {
            when (result) {
                is SolveResult.Sat -> {
                    put("verdict", "sat")
                    put("values", values(compiled, entries, result.assignment))
                }
                is SolveResult.Unsat -> put("verdict", "unsat")
                is SolveResult.Unknown -> {
                    put("verdict", "unknown")
                    put("reason", result.reason.toString())
                }
            }
        }
    }

    fun sample(request: ProblemQuery, control: OperationControl): JsonObject {
        val result = session.sample(params(request, control))
        control.checkpoint()
        return buildJsonObject {
            when (result) {
                is SampleResult.Found -> {
                    put("verdict", "sat")
                    put("values", values(compiled, entries, result.sample))
                }
                is SampleResult.Infeasible -> put("verdict", "unsat")
                is SampleResult.Unknown -> {
                    put("verdict", "unknown")
                    put("reason", result.reason.toString())
                }
            }
        }
    }

    private fun params(request: ProblemQuery, control: OperationControl) = BacktrackParams(
        maxDecisions = request.maxDecisions, randomSeed = request.seed,
        assumptions = assumptions(compiled, entries, request.assumptions), cancellation = control.cancellation,
    )

    override fun close() = session.close()
}

internal class StatisticsResource : ManagedResource {
    override val type = "kumulant.summary"
    private var stat = SummaryStat()
    private var count = 0L

    fun snapshot(): SummarySnapshot {
        val read = stat.read()
        return SummarySnapshot(count, read.mean, read.variance, read.min, read.max)
    }

    fun update(request: StatisticsUpdate, control: OperationControl): SummarySnapshot {
        val next = copyStat()
        for (value in request.values) {
            control.checkpoint()
            next.update(value)
        }
        return commit(next, count + request.values.size, control)
    }

    fun merge(snapshot: SummarySnapshot, control: OperationControl): SummarySnapshot {
        val next = copyStat()
        next.merge(SummaryResult(snapshot.count.toDouble(), snapshot.mean, snapshot.variance, snapshot.min, snapshot.max))
        return commit(next, count + snapshot.count, control)
    }

    fun reset(): SummarySnapshot {
        stat.reset()
        count = 0
        return snapshot()
    }

    private fun copyStat() = SummaryStat().also { it.merge(stat.read()) }

    private fun commit(next: SummaryStat, nextCount: Long, control: OperationControl): SummarySnapshot {
        val read = next.read()
        val snapshot = SummarySnapshot(nextCount, read.mean, read.variance, read.min, read.max)
        control.checkpoint()
        stat = next
        count = nextCount
        return snapshot
    }
}

internal class OptimizerResource(private val definition: CreateOptimizer) : ManagedResource {
    override val type = "combo.forest_ei"
    private val entries = definition.space.toSchemaDef().entries.also(::requireScalarVariables)
    private val space: CompiledDecisionSpace = compileInput { definition.space.compileSpace() }
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
        randomSeed = definition.seed,
        outcome = OutcomeSpec(maximize = definition.maximize),
    )

    fun describe(id: String): JsonObject = buildJsonObject {
        put("id", id)
        put("type", type)
        put("space", wireJson.encodeToJsonElement(definition.space))
        put("observationCount", optimizer.store.revision)
        put("pendingCount", optimizer.pendingTrials.size)
    }

    fun suggest(request: SuggestionRequest, control: OperationControl): JsonObject {
        val context = readContext(space, request.context)
        activeControl = control
        val batch = try { optimizer.suggest(request.count, context) } finally { activeControl = null }
        return buildJsonObject {
            put("termination", batch.termination.name)
            put("trials", buildJsonArray { batch.forEach { add(trialSnapshot(it)) } })
        }
    }

    fun trials() = buildJsonArray { optimizer.trials.forEach { add(trialSnapshot(it)) } }

    fun complete(id: Long, request: CompleteTrial): JsonObject {
        pending(id)
        optimizer.complete(id, request.reward, request.weight, request.noiseVariance)
        return trialSnapshot(trial(id))
    }

    fun fail(id: Long, reason: String): JsonObject {
        pending(id)
        optimizer.fail(id, reason)
        return trialSnapshot(trial(id))
    }

    fun cancel(id: Long): JsonObject {
        pending(id)
        optimizer.cancel(id)
        return trialSnapshot(trial(id))
    }

    private fun trial(id: Long): Trial = optimizer.trials.firstOrNull { it.id == id }
        ?: throw ApiException(HttpStatusCode.NotFound, "unknown_trial", "unknown trial")

    private fun pending(id: Long) {
        if (trial(id).status != TrialStatus.Pending) {
            throw ApiException(HttpStatusCode.Conflict, "trial_closed", "trial is already closed")
        }
    }

    private fun trialSnapshot(trial: Trial): JsonObject = buildJsonObject {
        put("id", trial.id)
        put("status", trial.status.name.lowercase())
        put("values", values(space.compiled, entries, trial.sample.sample))
        trial.reward?.let { put("reward", it) }
        trial.weight?.let { put("weight", it) }
        trial.noiseVariance?.let { put("noiseVariance", it) }
        trial.reason?.let { put("reason", it) }
    }
}

internal class ServerState(private val runner: OperationRunner = OperationRunner(), capacity: Int = 1000) : AutoCloseable {
    val problems = ResourceStore<ProblemResource>(runner, capacity)
    val statistics = ResourceStore<StatisticsResource>(runner, capacity)
    val optimizers = ResourceStore<OptimizerResource>(runner, capacity)

    override fun close() {
        problems.close()
        statistics.close()
        optimizers.close()
        runner.close()
    }
}
