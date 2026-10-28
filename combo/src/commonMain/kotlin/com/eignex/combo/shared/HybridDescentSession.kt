package com.eignex.combo.shared

import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.conditionedProblem
import com.eignex.combo.decisions.unframed
import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.backtrack.BacktrackSolver
import com.eignex.klause.ir.IntDomain
import com.eignex.klause.ir.randomValue
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
import com.eignex.klause.util.bigIntOf
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
    private val conditionedSearch: ((Context) -> WitnessSearch)? = null,
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
            assumptions.forEachInt { id, value -> if (witness.exactIntValue(id) != bigIntOf(value)) differences++ }
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
        if (request.context.hasRealValues) {
            val scoped = requireNotNull(conditionedSearch) { "this witness provider cannot condition real contexts" }
            return scoped(request.context).witnesses(
                request.copy(
                    context = Context.Empty,
                    anchors = request.anchors.filter { space.matches(it, request.context) },
                ),
                random,
            )
        }
        check(depth == 0) { "batch requests require an empty assumption stack" }
        val found = linkedMapOf<AssignmentKey, Sample>()
        val excluded = request.exclude.map { it.assignmentKey() }.toSet()
        fun capture(sample: Sample) {
            if (space.matches(sample, request.assumptions) && sample.assignmentKey() !in excluded) {
                found.getOrPut(sample.assignmentKey()) { sample }
            }
        }
        for (anchor in request.anchors) {
            if (space.matches(anchor, request.assumptions) && validator.valid(anchor)) {
                remember(anchor, request.assumptions)
            }
        }
        for (sample in pool) capture(sample)
        var attempts = 0
        var missed = false
        while ((found.size < request.count || attempts < candidateBatchSize) && attempts < request.maxAttempts) {
            attempts++
            // A stalled coordinate proposal may lie on an affine equality. Spend a bounded
            // attempt on joint geometry before asking the solver for the same endpoints again.
            if (space.coordinates.schema != null && missed && attempts % 2 == 0 && space.problem.numRealVars > 0) {
                val candidates = pool.filter { space.matches(it, request.assumptions) }
                val left = candidates.randomOrNull(random)
                val peers = if (left == null) {
                    emptyList()
                } else {
                    candidates.filter {
                        it.assignmentKey() != left.assignmentKey() && it.bools.contentEquals(left.bools) &&
                            it.ints.contentEquals(left.ints)
                    }
                }
                val right = peers.randomOrNull(random)
                val point = if (left == null || right == null) {
                    null
                } else {
                    interpolateWitness(left, right, random.nextDouble())
                }
                if (point != null && space.matches(point, request.assumptions) && validator.valid(point)) {
                    remember(point, request.assumptions)
                    capture(point)
                    lastResult = SampleResult.Found(point)
                }
                if (right != null) continue
            }
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
                    val before = found.size
                    capture(result.sample)
                    missed = found.size == before
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
        val key = sample.assignmentKey()
        pool.removeAll { it.assignmentKey() == key }
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
            context: Context = Context.Empty,
        ): HybridDescentSession {
            if (space.coordinates.schema == null) {
                return native(space, backtrack, candidateBatchSize, poolCapacity, context)
            }
            val problem = space.conditionedProblem(context).bake()
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
                        val sample = result.assignment?.let {
                            if (!context.hasRealValues) proposals.interior(it, rng) else it.unframed()
                        }
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
                        is SolveResult.Sat -> SampleResult.Found(result.assignment.unframed())
                        is SolveResult.Unsat -> SampleResult.Infeasible(result.core)
                        is SolveResult.Unknown -> SampleResult.Unknown(result.reason)
                    }
                },
                candidateBatchSize = candidateBatchSize,
                poolCapacity = poolCapacity,
                conditionedSearch = { given ->
                    klause(space, localSearch, backtrack, candidateBatchSize, poolCapacity, given)
                },
            )
        }

        private fun native(
            space: CompiledDecisionSpace,
            params: BacktrackParams,
            candidateBatchSize: Int,
            poolCapacity: Int,
            context: Context,
        ): HybridDescentSession {
            val model = space.conditionedProblem(context)
            val base = space.discreteAssumptionsFor(context)
            fun search(rng: Random, pins: Assumptions, randomize: Boolean): SampleResult {
                val objective = if (randomize) {
                    LinearObjective(
                        boolWeights = LongArray(model.numBoolVars) { if (rng.nextBoolean()) 1L else -1L },
                        intCoefficients = LongArray(model.numIntVars) { if (rng.nextBoolean()) 1L else -1L },
                        realCoefficients = DoubleArray(model.numRealVars) { rng.nextDouble(-1.0, 1.0) },
                    )
                } else {
                    LinearObjective()
                }
                val requested = base.mergedWith(pins)
                var proposed = requested
                if (randomize) {
                    for (id in 0 until model.numIntVars) {
                        if (id in requested.ints || id in context.exactInts) continue
                        val bounds = model.intBounds
                        if (!bounds.hasLower(id) || !bounds.hasUpper(id)) continue
                        val domain = model.intDomainOrNull(id) ?: IntDomain(bounds.lower(id), bounds.upper(id))
                        proposed = proposed.withInt(id, domain.randomValue(rng))
                    }
                }
                val result = model.sourceMinimize(
                    objective,
                    params.copy(randomSeed = rng.nextLong(), assumptions = proposed),
                ).sampleResult()
                if (result is SampleResult.Found) return result
                val fallback = if (proposed === requested) {
                    result
                } else {
                    model.sourceMinimize(
                        objective,
                        params.copy(randomSeed = rng.nextLong(), assumptions = requested),
                    ).sampleResult()
                }
                return if (fallback is SampleResult.Unknown) {
                    model.sourceSolve(
                        params.copy(randomSeed = rng.nextLong(), assumptions = requested),
                    ).sampleResult()
                } else {
                    fallback
                }
            }
            return HybridDescentSession(
                space,
                { rng, pins, _ -> search(rng, pins, true) },
                { rng, pins -> search(rng, pins, false) },
                candidateBatchSize,
                poolCapacity,
                conditionedSearch = { given -> native(space, params, candidateBatchSize, poolCapacity, given) },
            )
        }
    }
}
