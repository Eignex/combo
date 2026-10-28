package com.eignex.combo.bandit.dt

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.Context
import com.eignex.combo.shared.DescentSession
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.combo.shared.NoFeasibleSampleException
import com.eignex.combo.shared.PredictionLearner
import com.eignex.combo.shared.RegressionLearning
import com.eignex.combo.shared.integerObjective
import com.eignex.combo.util.RandomSequence
import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.localsearch.LocalSearchParams
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.propagation.PropagationResult
import com.eignex.klause.propagation.PropagationSession
import com.eignex.klause.solver.Sample
import com.eignex.klause.solver.objective.LinearObjective
import com.eignex.koblas.Vector
import com.eignex.kumulant.bandit.univariate.BanditPolicy
import com.eignex.kumulant.bandit.univariate.NormalTS
import com.eignex.kumulant.core.PairedStat
import com.eignex.kumulant.core.SeriesStat
import com.eignex.kumulant.stat.regression.tree.RandomForestRegressionStat
import com.eignex.kumulant.stat.regression.tree.RegressionLeafNode
import com.eignex.kumulant.stat.regression.tree.RegressionSplitNode
import com.eignex.kumulant.stat.regression.tree.RegressionTreeConfig
import com.eignex.kumulant.stat.regression.tree.aggregate
import com.eignex.kumulant.stat.regression.tree.mergeForestEvidence
import com.eignex.kumulant.stat.summary.WeightedVarianceResult
import kotlin.random.Random

/**
 * Online random forest bandit using **greedy literal selection with cross-tree
 * subtree merging** — the algorithm from the original com.eignex.combo. Each `choose` round:
 *
 *  1. Start with the caller's contextual variables pinned.
 *  2. For every tree, descend through splits whose direction is already pinned,
 *     stopping at the deepest unfixed [RegressionSplitNode] (the *frontier*).
 *  3. Group the frontier nodes by which klause variable they branch on — many
 *     trees can have the same variable still unfixed.
 *  4. For each candidate variable, **merge `pos.arm` and `neg.arm` across the
 *     trees that branch on it**, score both directions via `policy.evaluate`
 *     (one Thompson draw per direction when `policy` is `ThompsonSampling`), and
 *     pick the best `(variable, direction)` overall.
 *  5. Add that decision to the pinned set, repeat from step 2 until no frontier
 *     splits remain.
 *  6. Materialise: ask [proposeSample] for any feasible sample whose row routes
 *     to the chosen leaves under the assumption set.
 *
 * Per-round diversity sources still in play:
 *  - **Online bagging** (Oza & Russell 2001): per-tree Poisson(1) reweighting on
 *    every train. Trees see different sub-streams of the data.
 *  - **Per-leaf mtry**: each tree's audit leaves consider a random subset of
 *    candidate splits, picked at leaf birth from the tree's own RNG.
 *  - **Thompson per branching decision** at choose time, with the per-variable
 *    score derived from the *merged* per-tree subtree stats.
 *
 * Training: every observation visits every tree (weighted by bagging). Prediction:
 * pools per-tree leaf statistics at the queried sample. `optimalOrThrow`: same
 * iterative descent but with `Greedy` scoring (deterministic argmax at each step).
 *
 * After each greedy decision the bandit pushes the pin onto a
 * [com.eignex.klause.propagation.PropagationSession]: implied literals are folded into the
 * pin set (so future descents skip frontier splits on already-forced variables) and
 * over-constraint is caught immediately. On [PropagationResult.Unsat] the result's
 * [PropagationResult.Unsat.conflictLevels] names every decision level involved in the
 * conflict; the bandit calls [PropagationSession.popToLevel] with the deepest
 * conflicting level minus one — CDCL-style conflict-directed backjumping that skips
 * over any intermediate decisions not part of the conflict. The [LinearObjective]
 * fallback is now a last resort, kicking in only when the descent terminates with a
 * pin set the solver still can't materialise (a regime the propagator's
 * sound-but-incomplete proof of feasibility can't always detect upfront).
 *
 * The default builder adds a [HybridDescentSession]: candidate witnesses are scored
 * using one cached posterior draw per merged leaf tuple for the round. Boolean and nominal
 * predicates drive the original cross-tree descent; numeric and native float splits
 * contribute through complete-candidate scoring. A better candidate can remain the
 * incumbent even when it lies outside the last greedy branch. Context stays hard.
 */
class RandomForestBandit(
    val space: CompiledDecisionSpace,
    val policy: BanditPolicy<WeightedVarianceResult>,
    val proposeSample: (Random, Assumptions) -> Sample?,
    /** Kumulant model owning tree growth, online bagging, and state. */
    val regression: RandomForestRegressionStat,
    /** Numeric feature view and its inverse mapping to Klause predicates. */
    val layout: TreeVectorLayout,
    /**
     * Recovery hook for over-constrained decision sets. When [proposeSample] returns
     * null under the greedy-and-propagated pin set, the bandit builds a
     * [LinearObjective] from per-decision scores (positive coefficient pulls the
     * variable toward `false`, negative toward `true`; magnitude is the policy
     * score) and asks this lambda to minimise it. Greedy decision pins become soft
     * preferences; the caller's context remains a hard scope. Null disables this hook;
     * failure to find a witness raises [NoFeasibleSampleException], which need not mean UNSAT.
     */
    val optimizeFallback: ((LinearObjective, Assumptions) -> Sample?)? = null,
    /**
     * klause session-based incremental solver driving the *primary* descent. When
     * present, `choose` first runs the iterative session descent — pinning decisions
     * onto the session's assumption stack and committing only those that keep a witness
     * solvable. When the base problem can't be solved iteratively (a very hard instance
     * the backend can't crack incrementally), the bandit falls back to the
     * propagation-based CDCL descent below. Null → always use the CDCL descent.
     */
    val descentSession: DescentSession? = null,
    override val randomSeed: Int = System.currentTimeMillis().toInt(),
    override val maximize: Boolean = true,
    override val rewards: SeriesStat<*>? = null,
    override val trainScore: PairedStat<*>? = null,
    override val testScore: PairedStat<*>? = null,
) : PredictionLearner<ForestData> {

    init {
        require(regression.featureSize == layout.featureSize) { "forest and feature layout sizes differ" }
        require(regression.splitCandidates == layout.splits) { "forest and feature layout predicates differ" }
    }

    /** Whether the underlying model applies online bagging. */
    val bagging: Boolean get() = regression.bagging

    private val randomSequence = RandomSequence(randomSeed)

    /** Live typed views of the trees owned by [regression]. */
    val trees: List<Tree> = List(regression.nbrTrees) { index -> Tree(policy, layout) { regression.tree(index) } }
    private val projection = TreeFeatureProjection(space)
    private val boolIdByName = space.coordinates.boolVarIdByName
    private val intIdByName = space.coordinates.intVarIdByName
    private val nominalIndicators = space.coordinates.nominalIndicators
    private var step: Long = 0L

    override fun chooseOrThrow(): BanditSample = chooseOrThrow(Context.Empty)
    override fun optimalOrThrow(): BanditSample = optimalOrThrow(Context.Empty)

    /** Choose an exploratory configuration with every contextual variable pinned. */
    fun chooseOrThrow(context: Context): BanditSample = descend(context, thompson = true)

    /** Choose with context, returning null when no witness is found within the search budget. */
    fun choose(context: Context): BanditSample? = try {
        chooseOrThrow(context)
    } catch (_: NoFeasibleSampleException) {
        null
    }

    /** Recommend a configuration using posterior means with the supplied context pinned. */
    fun optimalOrThrow(context: Context): BanditSample = descend(context, thompson = false)

    /** Recommend with context, returning null when the search cannot produce a witness. */
    fun optimal(context: Context): BanditSample? = try {
        optimalOrThrow(context)
    } catch (_: NoFeasibleSampleException) {
        null
    }

    /**
     * Primary path: klause session-based iterative descent (when a [descentSession] is
     * wired). Falls back to the propagation-based CDCL descent for hard problems the
     * session solver can't crack iteratively (signalled by a null return).
     */
    private fun descend(context: Context, thompson: Boolean): BanditSample {
        val assumptions = space.assumptionsFor(context)
        val rng = randomSequence.next()
        val t = if (thompson) step++ else step
        val sess = descentSession
        if (sess != null) {
            if (sess is HybridDescentSession) sess.beginRound(candidateScorer(thompson, rng, t))
            greedyDescentSession(sess, thompson, rng, t, assumptions)?.let { return it }
        }
        return greedyDescentCdcl(thompson, rng, t, assumptions)
    }

    private fun candidateScorer(thompson: Boolean, rng: Random, t: Long): (Sample) -> Double {
        val draws = mutableMapOf<List<RegressionLeafNode<Vector>>, Double>()
        return { sample ->
            val row = projection.encode(BanditSample(sample))
            val leaves = regression.findLeaves(layout.encode(row))
            draws.getOrPut(leaves) {
                val snapshot = mergeForestEvidence(leaves.map { it.arm.read(0L) })
                signed(if (thompson) policy.evaluate(snapshot, t, rng) else snapshot.mean)
            }
        }
    }

    /**
     * Session-based greedy descent. Pins each chosen `(variable, direction)` onto the
     * klause [DescentSession] stack and keeps it only when a witness is still solvable;
     * an unresolved or infeasible pin is popped and recorded as attempted this round.
     * Because every kept pin
     * is verified feasible, the stack stays satisfiable throughout — no backjumping is
     * needed. Returns null (so the caller falls back to CDCL) when even the base problem
     * yields no witness from the iterative solver.
     */
    private fun greedyDescentSession(
        sess: DescentSession,
        thompson: Boolean,
        rng: Random,
        t: Long,
        assumptions: Assumptions,
    ): BanditSample? {
        while (sess.depth > 0) sess.pop()
        sess.push(assumptions)
        try {
            val base = sess.sample(rng) ?: return null
            var witness: Sample = base
            val pinned = assumptions.bools.toMutableMap()
            val tried = mutableSetOf<Pair<Int, Boolean>>()
            while (true) {
                val byVar = mutableMapOf<Int, MutableList<RegressionSplitNode<Vector>>>()
                for (tree in trees) {
                    val node = tree.descendTo(
                        pinned,
                        boolIdByName,
                        space.coordinates.nominalIndicators,
                        assumptions,
                        space.coordinates.intVarIdByName,
                    )
                    if (node !is RegressionSplitNode) continue
                    val split = layout.predicate(node.split)
                    val id = split.booleanId(boolIdByName, nominalIndicators) ?: continue
                    if (id in pinned) continue
                    byVar.getOrPut(id) { mutableListOf() }.add(node)
                }
                if (byVar.isEmpty()) break

                var bestId = -1
                var bestDirection = true
                var bestScore = Double.NEGATIVE_INFINITY
                for ((id, nodes) in byVar) {
                    val posSnap = mergeForestEvidence(nodes.map { it.pos.aggregate() })
                    val negSnap = mergeForestEvidence(nodes.map { it.neg.aggregate() })
                    val sPos = if (thompson) signed(policy.evaluate(posSnap, t, rng)) else signed(posSnap.mean)
                    val sNeg = if (thompson) signed(policy.evaluate(negSnap, t, rng)) else signed(negSnap.mean)
                    if (id to true !in tried &&
                        sPos > bestScore
                    ) {
                        bestId = id
                        bestDirection = true
                        bestScore = sPos
                    }
                    if (id to false !in tried &&
                        sNeg > bestScore
                    ) {
                        bestId = id
                        bestDirection = false
                        bestScore = sNeg
                    }
                }
                if (bestId < 0) break

                // Keep only witnessed branches. A budget miss is an attempted direction,
                // not an infeasibility proof; retain the previous feasible incumbent.
                sess.push(Assumptions(bools = mapOf(bestId to bestDirection)))
                val s = sess.sample(rng)
                if (s != null) {
                    pinned[bestId] = bestDirection
                    witness = s
                } else {
                    sess.pop()
                    tried += bestId to bestDirection
                }
            }
            // Failed or unproductive branch repairs never discard a better feasible incumbent.
            return BanditSample((sess as? HybridDescentSession)?.bestWitness(assumptions, rng) ?: witness)
        } finally {
            while (sess.depth > 0) sess.pop()
        }
    }

    /**
     * Iterative greedy literal-selection loop. When [thompson] is true, scores each
     * candidate `(variable, direction)` via [policy] (which does fresh Thompson draws
     * if it's `ThompsonSampling`); when false, scores by deterministic mean — the
     * exploit-only counterpart used by `optimalOrThrow`.
     */
    private fun greedyDescentCdcl(thompson: Boolean, rng: Random, t: Long, assumptions: Assumptions): BanditSample {
        // Stateful propagation: push pins as we decide; the session's incremental
        // engine handles fixpoint reuse across pushes and snapshot-restore on pop.
        // Seeding with context pins also triggers constraint propagation,
        // so any literal forced by the problem alone joins the pin set before any
        // greedy decision is made.
        val session = PropagationSession(space.coordinates.problem)
        if (session.seed(assumptions) is PropagationResult.Unsat) {
            throw NoFeasibleSampleException("decision space is infeasible under the supplied context")
        }

        // Per-decision-level scores and directions. The variable id at level `L` is
        // recoverable from `session.decisionAt(L)`; the session is the source of truth
        // for the trail. These two parallel lists carry only what klause can't tell
        // us — the bandit's chosen direction and the policy score that drove it.
        // Both lists are indexed by `level - 1`.
        val decisionDirs = mutableListOf<Boolean>()
        val decisionScores = mutableListOf<Double>()
        // (varId, direction) pairs we've already attempted (or backtracked through) —
        // never reattempt the same direction at the same variable in this round.
        val tried = mutableSetOf<Pair<Int, Boolean>>()

        descent@ while (true) {
            val pinned: Map<Int, Boolean> = session.currentAssumptions().bools

            // Group frontier SplitNodes by the klause bool var they branch on. Variables
            // already pinned (whether by a decision or by implication) aren't on any
            // tree's frontier — descendTo walked past them.
            val byVar = mutableMapOf<Int, MutableList<RegressionSplitNode<Vector>>>()
            for (tree in trees) {
                val node = tree.descendTo(
                    pinned,
                    boolIdByName,
                    space.coordinates.nominalIndicators,
                    assumptions,
                    space.coordinates.intVarIdByName,
                )
                if (node !is RegressionSplitNode) continue
                val split = layout.predicate(node.split)
                val id = split.booleanId(boolIdByName, nominalIndicators) ?: continue
                byVar.getOrPut(id) { mutableListOf() }.add(node)
            }
            if (byVar.isEmpty()) break

            // For each candidate variable, merge pos/neg subtree aggregates across the
            // trees whose frontier branches on it; score both directions, pick the best
            // (id, direction) pair that hasn't already been tried.
            var bestId = -1
            var bestDirection = true
            var bestScore = Double.NEGATIVE_INFINITY
            for ((id, nodes) in byVar) {
                val posSnap = mergeForestEvidence(nodes.map { it.pos.aggregate() })
                val negSnap = mergeForestEvidence(nodes.map { it.neg.aggregate() })
                val sPos = if (thompson) {
                    signed(policy.evaluate(posSnap, t, rng))
                } else {
                    signed(posSnap.mean)
                }
                val sNeg = if (thompson) {
                    signed(policy.evaluate(negSnap, t, rng))
                } else {
                    signed(negSnap.mean)
                }
                if (id to true !in tried && sPos > bestScore) {
                    bestId = id
                    bestDirection = true
                    bestScore = sPos
                }
                if (id to false !in tried && sNeg > bestScore) {
                    bestId = id
                    bestDirection = false
                    bestScore = sNeg
                }
            }
            // Every frontier candidate exhausted by `tried` → terminate descent.
            if (bestId < 0) break

            when (val result = session.pinBool(bestId, bestDirection)) {
                is PropagationResult.Implied -> {
                    decisionDirs += bestDirection
                    decisionScores += bestScore
                }

                is PropagationResult.Unsat -> {
                    // Session auto-restored to pre-push state on Unsat — no popLast needed.
                    tried += bestId to bestDirection

                    // Conflict-directed backjump: pop past the deepest conflicting
                    // decision. `conflictLevels` may contain levels that aren't the
                    // most recent; popping to `max - 1` skips over irrelevant in-between
                    // decisions in one move (proper CDCL backjump). When the conflict
                    // is rooted in problem constraints alone — no decision involved —
                    // there's nothing to backjump to and the loop just picks a different
                    // candidate next iteration.
                    val deepest = result.conflictLevels.maxOrNull()?.takeIf { it >= 1 }
                    if (deepest != null) {
                        val (_, popVar) = session.decisionAt(deepest)
                            ?: continue@descent
                        val popDir = decisionDirs[deepest - 1]
                        session.popToLevel(deepest - 1)
                        tried += popVar to popDir
                        while (decisionDirs.size > deepest - 1) {
                            decisionDirs.removeAt(decisionDirs.size - 1)
                            decisionScores.removeAt(decisionScores.size - 1)
                        }
                    }
                }
            }
        }

        // Happy path: pin set is satisfiable; klause finds a witness.
        val raw = proposeSample(rng, session.currentAssumptions())
        if (raw != null) return BanditSample(raw)

        // Fallback path: soft-satisfy via LinearObjective.
        return runFallback(session, decisionDirs, decisionScores, assumptions)
    }

    private fun runFallback(
        session: PropagationSession,
        decisionDirs: List<Boolean>,
        decisionScores: List<Double>,
        assumptions: Assumptions,
    ): BanditSample {
        val fallback = optimizeFallback ?: throw NoFeasibleSampleException(
            "RandomForestBandit could not materialize its selected decisions " +
                "(${decisionDirs.size} greedy pins) and no optimizeFallback was supplied.",
        )
        // Reconstruct the per-decision variable ids from the session trail (decisions
        // are at levels 1..decisionDirs.size, each a Bool decision in combo's setup).
        val ids = IntArray(decisionDirs.size) { i ->
            session.decisionAt(i + 1)?.second
                ?: error("session decision at level ${i + 1} missing during fallback build")
        }
        val objective = buildSoftObjective(ids.toList(), decisionDirs, decisionScores)
        val raw = fallback(objective, assumptions)
            ?: throw NoFeasibleSampleException(
                "RandomForestBandit fallback could not find any feasible sample.",
            )
        return BanditSample(raw)
    }

    /**
     * Build a [LinearObjective] whose minimum is the feasible sample that best matches
     * the bandit's greedy decisions weighted by their scores. Klause minimises, so:
     *   - decision (id, value=true) with score s → `boolWeights[id]` -= s (lower cost when true)
     *   - decision (id, value=false) with score s → `boolWeights[id]` += s (lower cost when false)
     * The sign of `s` already reflects the maximise/minimise direction from policy scoring.
     */
    private fun buildSoftObjective(ids: List<Int>, dirs: List<Boolean>, scores: List<Double>): LinearObjective {
        val n = space.coordinates.problem.numBoolVars
        val weights = DoubleArray(n)
        for (i in ids.indices) {
            val id = ids[i]
            val s = scores[i]
            if (dirs[i]) weights[id] -= s else weights[id] += s
        }
        return integerObjective(space.coordinates.problem, boolWeights = weights)
    }

    private fun signed(m: Double): Double = if (maximize) m else -m

    private val learning = RegressionLearning(
        regression,
        { sample, _ -> layout.encode(projection.encode(sample)) },
        regression::predict,
        rewards,
        trainScore,
        testScore,
    )

    override fun predict(sample: BanditSample): Double = learning.predict(sample)

    override fun update(sample: BanditSample, reward: Double, weight: Double) =
        learning.update(sample, Context.Empty, reward, weight)

    override fun train(sample: BanditSample, reward: Double, weight: Double) =
        learning.train(sample, Context.Empty, reward, weight)

    /** Predict after verifying that the sample contains the caller's context. */
    fun predict(sample: BanditSample, context: Context): Double {
        requireContext(sample, context)
        return predict(sample)
    }

    /** Train on a configuration whose contextual variables match the supplied values. */
    fun train(sample: BanditSample, context: Context, reward: Double, weight: Double = 1.0) {
        requireContext(sample, context)
        train(sample, reward, weight)
    }

    /** Record a contextual observation and update its forest posteriors. */
    fun update(sample: BanditSample, context: Context, reward: Double, weight: Double = 1.0) {
        requireContext(sample, context)
        update(sample, reward, weight)
    }

    private fun requireContext(sample: BanditSample, context: Context) {
        require(space.matches(sample.sample, space.assumptionsFor(context))) {
            "sample contextual values do not match the supplied context"
        }
    }

    override fun importData(data: ForestData) {
        require(
            layout.matches(data.splits),
        ) { "forest split layouts differ" }
        learning.merge(data.state)
    }
    override fun exportData(): ForestData = ForestData(learning.snapshot(), layout.splits.map(layout::predicate))

    companion object {
        /** Build a forest using a witness pool, local-search repairs, and bounded Klause backtracking. */
        fun build(
            space: CompiledDecisionSpace,
            policy: BanditPolicy<WeightedVarianceResult> = NormalTS(),
            nbrTrees: Int = 10,
            mtry: Int? = null,
            config: RegressionTreeConfig = RegressionTreeConfig(),
            localSearch: LocalSearchParams = LocalSearchParams(maxFlips = 2_000L),
            backtrack: BacktrackParams = BacktrackParams(maxDecisions = 10_000L),
            candidateBatchSize: Int = 8,
            poolCapacity: Int = 64,
            randomSeed: Int = System.currentTimeMillis().toInt(),
            maximize: Boolean = true,
            rewards: SeriesStat<*>? = null,
            trainScore: PairedStat<*>? = null,
            testScore: PairedStat<*>? = null,
            bagging: Boolean = true,
            candidates: List<Split> = defaultSplitCandidates(space),
        ): RandomForestBandit {
            val hybrid = HybridDescentSession.klause(space, localSearch, backtrack, candidateBatchSize, poolCapacity)
            return build(
                space = space,
                policy = policy,
                bagging = bagging,
                candidates = candidates,
                proposeSample = { rng, assumptions -> hybrid.search(rng, assumptions).assignment },
                nbrTrees = nbrTrees,
                mtry = mtry,
                config = config,
                descentSession = hybrid,
                randomSeed = randomSeed,
                maximize = maximize,
                rewards = rewards,
                trainScore = trainScore,
                testScore = testScore,
            )
        }

        /**
         * Convenience constructor: builds [nbrTrees] trees over the [space]'s default
         * split-candidate pool, each with its own per-leaf mtry subspace (the
         * Breiman default `⌈√p⌉` unless overridden). Seeded deterministically from
         * [randomSeed] so the same call always produces the same forest.
         */
        fun build(
            space: CompiledDecisionSpace,
            policy: BanditPolicy<WeightedVarianceResult>,
            proposeSample: (Random, Assumptions) -> Sample?,
            nbrTrees: Int = 10,
            mtry: Int? = null,
            config: RegressionTreeConfig = RegressionTreeConfig(),
            optimizeFallback: ((LinearObjective, Assumptions) -> Sample?)? = null,
            descentSession: DescentSession? = null,
            randomSeed: Int = System.currentTimeMillis().toInt(),
            maximize: Boolean = true,
            rewards: SeriesStat<*>? = null,
            trainScore: PairedStat<*>? = null,
            testScore: PairedStat<*>? = null,
            bagging: Boolean = true,
            candidates: List<Split> = defaultSplitCandidates(space),
        ): RandomForestBandit {
            val layout = TreeVectorLayout(candidates)
            val regression = RandomForestRegressionStat(
                featureSize = layout.featureSize,
                splitCandidates = layout.splits,
                nbrTrees = nbrTrees,
                config = config.copy(mtry = mtry ?: config.mtry),
                bagging = bagging,
                leafArmFactory = { policy.createArm() },
                randomSeed = randomSeed,
            )
            return RandomForestBandit(
                space = space,
                policy = policy,
                proposeSample = proposeSample,
                regression = regression,
                layout = layout,
                optimizeFallback = optimizeFallback,
                descentSession = descentSession,
                randomSeed = randomSeed,
                maximize = maximize,
                rewards = rewards,
                trainScore = trainScore,
                testScore = testScore,
            )
        }
    }
}
