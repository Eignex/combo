package com.eignex.combo.shared

import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.backtrack.BacktrackSolver
import com.eignex.klause.localsearch.LocalSearchParams
import com.eignex.klause.localsearch.LocalSearchSession
import com.eignex.klause.localsearch.LocalSearchSolver
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.propagation.bake
import com.eignex.klause.solver.Sample
import com.eignex.klause.solver.SolveResult
import com.eignex.klause.solver.objective.LinearObjective
import com.eignex.klause.solver.result.MinimizeResult
import com.eignex.klause.solver.result.SampleResult
import com.eignex.klause.solver.result.TerminationReason
import kotlin.random.Random

/**
 * Single-threaded constrained search retaining a bounded pool of solver-produced witnesses.
 * Each round refreshes the pool, ranks compatible witnesses using a fixed scoring
 * function, and repairs a nearby witness only when no cached one fits a branch.
 * A failed repair escalates to bounded backtracking. [lastResult] preserves the
 * distinction between an infeasibility proof and an exhausted search budget.
 *
 * Both callbacks must return witnesses verified by their solver under the supplied
 * assumptions. The compiled decision space must remain unchanged for the lifetime of this session.
 */
class HybridDescentSession(
    private val space: CompiledDecisionSpace,
    private val propose: (Random, Assumptions, Sample?) -> SampleResult,
    private val backtrack: (Random, Assumptions) -> SampleResult,
    private val candidateBatchSize: Int = 8,
    private val poolCapacity: Int = 64,
) : DescentSession,
    WitnessSearch {
    private val validator = WitnessValidator(space)
    private val scopes = mutableListOf<Assumptions>()
    private val pool = ArrayDeque<Sample>()
    private var score: (Sample) -> Double = { 0.0 }
    private var refresh = false

    /** Most recent pool lookup or solver result, including Unknown on budget exhaustion. */
    var lastResult: SampleResult? = null
        private set

    /** Number of requests answered using an already-feasible witness. */
    var poolHits: Long = 0
        private set

    /** Number of fresh candidate and warm-start repair requests. */
    var proposalCalls: Long = 0
        private set

    /** Number of escalations to the bounded backtracking callback. */
    var backtrackCalls: Long = 0
        private set

    override val depth: Int get() = scopes.size

    init {
        require(candidateBatchSize > 0) { "candidateBatchSize must be positive" }
        require(poolCapacity >= candidateBatchSize) { "poolCapacity must cover a candidate batch" }
    }

    /** Set the fixed scorer for a round and request a fresh candidate batch. */
    fun beginRound(scoreSample: (Sample) -> Double) {
        check(depth == 0) { "a hybrid round requires an empty assumption stack" }
        score = scoreSample
        refresh = true
    }

    override fun push(assumptions: Assumptions) {
        scopes += (scopes.lastOrNull() ?: Assumptions.None).mergedWith(assumptions)
    }

    override fun pop() {
        check(scopes.isNotEmpty()) { "cannot pop an empty hybrid session" }
        scopes.removeAt(scopes.lastIndex)
    }

    override fun sample(rng: Random): Sample? = search(rng, scopes.lastOrNull() ?: Assumptions.None).assignment

    /** Search outside the session stack, preserving the caller's hard assumptions. */
    fun search(rng: Random, assumptions: Assumptions): SampleResult {
        if (refresh) {
            refresh = false
            repeat(candidateBatchSize) {
                proposalCalls++
                val candidate = propose(rng, assumptions, null)
                if (candidate is SampleResult.Found) remember(candidate.sample, assumptions)
            }
        }
        bestWitness(assumptions, rng)?.let {
            poolHits++
            return SampleResult.Found(it).also { result -> lastResult = result }
        }
        val anchor = pool.minByOrNull { witness ->
            var differences = 0
            assumptions.forEachBool { id, value -> if (witness.boolValue(id) != value) differences++ }
            assumptions.forEachInt { id, value -> if (witness.intValue(id) != value) differences++ }
            differences
        }
        proposalCalls++
        val repaired = propose(rng, assumptions, anchor)
        val result = if (repaired is SampleResult.Unknown) {
            backtrackCalls++
            backtrack(rng, assumptions)
        } else {
            repaired
        }
        if (result is SampleResult.Found) remember(result.sample, assumptions)
        lastResult = result
        return result
    }

    override fun witnesses(request: WitnessRequest, random: Random): WitnessBatch {
        check(depth == 0) { "batch requests require an empty assumption stack" }
        val found = linkedMapOf<AssignmentKey, Sample>()
        val excluded = request.exclude.map { it.assignmentKey() }.toSet()
        fun capture(sample: Sample) {
            if (space.matches(sample, request.assumptions) && sample.assignmentKey() !in excluded) {
                found.putIfAbsent(sample.assignmentKey(), sample)
            }
        }
        for (anchor in request.anchors) {
            if (space.matches(anchor, request.assumptions) && validator.valid(anchor)) {
                remember(anchor, request.assumptions)
            }
        }
        for (sample in pool) capture(sample)
        var attempts = 0
        while ((found.size < request.count || attempts < candidateBatchSize) && attempts < request.maxAttempts) {
            attempts++
            proposalCalls++
            var result = propose(random, request.assumptions, null)
            if (result is SampleResult.Unknown && attempts < request.maxAttempts) {
                attempts++
                backtrackCalls++
                result = backtrack(random, request.assumptions)
            }
            lastResult = result
            when (result) {
                is SampleResult.Found -> {
                    remember(result.sample, request.assumptions)
                    capture(result.sample)
                }

                is SampleResult.Infeasible -> if (found.isEmpty()) {
                    return WitnessBatch(emptyList(), WitnessTermination.Infeasible)
                }

                is SampleResult.Unknown -> Unit
            }
        }
        val termination = if (found.size >= request.count) {
            WitnessTermination.CountReached
        } else {
            WitnessTermination.BudgetExhausted
        }
        return WitnessBatch(found.values.shuffled(random).take(request.count), termination)
    }

    internal fun bestWitness(assumptions: Assumptions, rng: Random): Sample? =
        pool.filter { space.matches(it, assumptions) }.shuffled(rng).maxByOrNull(score)

    private fun remember(sample: Sample, assumptions: Assumptions) {
        check(space.matches(sample, assumptions)) { "solver returned a witness outside the requested scope" }
        pool.remove(sample)
        pool.addLast(sample)
        if (pool.size > poolCapacity) pool.removeFirst()
    }

    /** Factories connecting witness reuse to Klause's search engines. */
    companion object {
        /** Create a local-search repair session with bounded backtracking as its fallback. */
        fun klause(
            space: CompiledDecisionSpace,
            localSearch: LocalSearchParams = LocalSearchParams(maxFlips = 2_000L),
            backtrack: BacktrackParams = BacktrackParams(maxDecisions = 10_000L),
            candidateBatchSize: Int = 8,
            poolCapacity: Int = 64,
        ): HybridDescentSession {
            val problem = space.compiled.problem.bake()
            val local = LocalSearchSession(LocalSearchSolver(problem))
            val complete = BacktrackSolver(problem)
            val proposals = ScaledWitnessProposals(space)
            return HybridDescentSession(
                space = space,
                propose = { rng, assumptions, anchor ->
                    if (anchor == null && problem.numRealVars > 0) {
                        // Solver completions seed independently verified interior proposals.
                        // Failed proposals retain the feasible solver witness.
                        val objective = LinearObjective(
                            LongArray(problem.numBoolVars) { if (rng.nextBoolean()) 1L else -1L },
                            realCoefficients = DoubleArray(problem.numRealVars) { rng.nextDouble(-1.0, 1.0) },
                        )
                        val pins = proposals.integerPins(assumptions, rng)
                        val proposed = complete.minimize(
                            objective,
                            backtrack.copy(randomSeed = rng.nextLong(), assumptions = pins),
                        )
                        val result = if (proposed.assignment == null && pins !== assumptions) {
                            complete.minimize(
                                objective,
                                backtrack.copy(randomSeed = rng.nextLong(), assumptions = assumptions),
                            )
                        } else {
                            proposed
                        }
                        val sample = result.assignment?.let { proposals.interior(it, rng) }
                        when {
                            sample != null -> SampleResult.Found(sample)
                            result is MinimizeResult.Infeasible -> SampleResult.Infeasible(result.core)
                            result is MinimizeResult.Unknown -> SampleResult.Unknown(result.reason)
                            else -> SampleResult.Unknown(TerminationReason.BudgetExhausted)
                        }
                    } else {
                        val pins = proposals.integerPins(assumptions, rng)
                        val proposed = local.sample(
                            localSearch.copy(
                                randomSeed = rng.nextLong(),
                                assumptions = pins,
                                initialAssignment = anchor ?: localSearch.initialAssignment,
                            ),
                        )
                        if (proposed is SampleResult.Found || pins === assumptions) {
                            proposed
                        } else {
                            local.sample(localSearch.copy(randomSeed = rng.nextLong(), assumptions = assumptions))
                        }
                    }
                },
                backtrack = { rng, assumptions ->
                    when (
                        val result = complete.solve(
                            backtrack.copy(randomSeed = rng.nextLong(), assumptions = assumptions),
                        )
                    ) {
                        is SolveResult.Sat -> SampleResult.Found(result.assignment)
                        is SolveResult.Unsat -> SampleResult.Infeasible(result.core)
                        is SolveResult.Unknown -> SampleResult.Unknown(result.reason)
                    }
                },
                candidateBatchSize = candidateBatchSize,
                poolCapacity = poolCapacity,
            )
        }
    }
}
