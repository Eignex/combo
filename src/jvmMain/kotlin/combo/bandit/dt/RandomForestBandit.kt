package combo.bandit.dt

import com.eignex.klause.solver.Assumptions
import com.eignex.klause.solver.LinearObjective
import com.eignex.klause.solver.PropagationResult
import com.eignex.klause.solver.PropagationSession
import com.eignex.klause.solver.Sample
import com.eignex.klause.solver.VarKind
import com.eignex.kumulant.core.Result
import com.eignex.kumulant.core.SeriesStat
import combo.bandit.NoFeasibleSampleException
import combo.bandit.PredictionBandit
import combo.bandit.univariate.BanditPolicy
import combo.decisions.BanditSample
import combo.decisions.CompiledDecisionSpace
import combo.util.RandomSequence
import kotlin.math.abs
import kotlin.math.exp
import kotlin.random.Random

/**
 * Online random forest bandit using **greedy literal selection with cross-tree
 * subtree merging** — the algorithm from the original combo. Each `choose` round:
 *
 *  1. Maintain a set of pinned bool decisions, starting empty.
 *  2. For every tree, descend through splits whose direction is already pinned,
 *     stopping at the deepest unfixed [SplitNode] (the *frontier*).
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
 * averages per-tree leaf means at the queried sample. `optimalOrThrow`: same
 * iterative descent but with `Greedy` scoring (deterministic argmax at each step).
 *
 * After each greedy decision the bandit pushes the pin onto a
 * [com.eignex.klause.solver.PropagationSession]: implied literals are folded into the
 * pin set (so future descents skip frontier splits on already-forced variables) and
 * over-constraint is caught immediately. On [PropagationResult.Unsat] the conflict set
 * identifies which prior decisions are jointly bad; the bandit pops the most-recent
 * decision in the conflict and tries a different one — CDCL-style conflict-directed
 * backjumping. The [LinearObjective] fallback is now a last resort, kicking in only
 * when the descent terminates with a pin set the solver still can't materialise (a
 * regime the propagator's sound-but-incomplete proof of feasibility can't always
 * detect upfront).
 *
 * Slice limitations: only [BoolSplit] frontier splits are scored; other split types
 * stop the descent on the tree carrying them. Adding [NominalSplit] / float-typed
 * frontier scoring is a small future slice.
 */
class RandomForestBandit<R : Result>(
    val space: CompiledDecisionSpace,
    val policy: BanditPolicy<R>,
    val proposeSample: (Random, Assumptions) -> Sample?,
    val trees: List<Tree<R>>,
    val retryBudget: Int = 32,
    /**
     * Recovery hook for over-constrained decision sets. When [proposeSample] returns
     * null under the greedy-and-propagated pin set, the bandit builds a
     * [LinearObjective] from per-decision scores (positive coefficient pulls the
     * variable toward `false`, negative toward `true`; magnitude is the policy
     * score) and asks this lambda to minimise it. The fallback runs with no hard
     * assumptions — any feasible sample is acceptable, the objective just
     * prefers ones that satisfy the bandit's collective vote. Null disables the
     * fallback; the bandit then throws [NoFeasibleSampleException] on
     * over-constraint.
     */
    val optimizeFallback: ((LinearObjective, Assumptions) -> Sample?)? = null,
    /** Enable Oza-Russell online bagging: per-tree Poisson(1) reweighting on every
     *  training observation. Defaults to true; turn off to make every tree see every
     *  observation identically (only useful when trees diverge solely via mtry). */
    val bagging: Boolean = true,
    override val randomSeed: Int = System.currentTimeMillis().toInt(),
    override val maximize: Boolean = true,
    override val rewards: SeriesStat<*>? = null,
    override val trainAbsError: SeriesStat<*>? = null,
    override val testAbsError: SeriesStat<*>? = null,
) : PredictionBandit<ForestData> {

    init {
        require(trees.isNotEmpty()) { "RandomForestBandit needs at least one tree" }
    }

    private val randomSequence = RandomSequence(randomSeed)
    private val baggingRng = Random(randomSeed.toLong() xor 0x9e3779b97f4a7c15uL.toLong())
    private val projection = TreeFeatureProjection(space)
    private var step: Long = 0L

    override fun chooseOrThrow(): BanditSample = greedyDescent(thompson = true)
    override fun optimalOrThrow(): BanditSample = greedyDescent(thompson = false)

    /**
     * Iterative greedy literal-selection loop. When [thompson] is true, scores each
     * candidate `(variable, direction)` via [policy] (which does fresh Thompson draws
     * if it's `ThompsonSampling`); when false, scores by deterministic mean — the
     * exploit-only counterpart used by `optimalOrThrow`.
     */
    private fun greedyDescent(thompson: Boolean): BanditSample {
        val rng = randomSequence.next()
        val t = if (thompson) step++ else step
        val boolIdByName = space.compiled.boolVarIdByName

        // Stateful propagation: push pins as we decide, pop on Unsat. Seeding with
        // Assumptions.None lets the session capture any literals forced by problem
        // constraints alone before any greedy decision is made.
        val session = PropagationSession(space.compiled.problem)
        if (session.seed(Assumptions.None) is PropagationResult.Unsat) {
            return runFallback(rng, emptyList(), emptyList(), emptyList())
        }

        // Tracks just the *greedy* decisions (not propagated consequences). Drives
        // both conflict-directed backtracking and the [LinearObjective] fallback.
        val decisionIds = mutableListOf<Int>()
        val decisionDirs = mutableListOf<Boolean>()
        val decisionScores = mutableListOf<Double>()
        // (varId, direction) pairs we already attempted (or backtracked through) —
        // never try the same direction at the same variable again this round.
        val tried = mutableSetOf<Pair<Int, Boolean>>()

        descent@ while (true) {
            val pinned: Map<Int, Boolean> = session.currentAssumptions().bools

            // Group frontier SplitNodes by the klause bool var they branch on. Variables
            // already pinned (whether by a decision or by implication) aren't on any
            // tree's frontier — descendTo walked past them.
            val byVar = mutableMapOf<Int, MutableList<SplitNode<R>>>()
            for (tree in trees) {
                val node = tree.descendTo(pinned, boolIdByName)
                if (node !is SplitNode) continue
                val split = node.split
                if (split !is BoolSplit) continue
                val id = boolIdByName[split.handle.name] ?: continue
                byVar.getOrPut(id) { mutableListOf() }.add(node)
            }
            if (byVar.isEmpty()) break

            // For each candidate variable, merge pos/neg arms across the trees whose
            // frontier branches on it; score both directions, pick the best
            // (id, direction) pair that hasn't already been tried.
            var bestId = -1
            var bestDirection = true
            var bestScore = Double.NEGATIVE_INFINITY
            for ((id, nodes) in byVar) {
                val merger = trees[0]
                val posSnap = merger.mergeArms(nodes.map { it.pos.arm })
                val negSnap = merger.mergeArms(nodes.map { it.neg.arm })
                val sPos = if (thompson) policy.evaluate(posSnap, t, maximize, rng)
                           else signed(scalarMean(posSnap))
                val sNeg = if (thompson) policy.evaluate(negSnap, t, maximize, rng)
                           else signed(scalarMean(negSnap))
                if (id to true !in tried && sPos > bestScore) { bestId = id; bestDirection = true; bestScore = sPos }
                if (id to false !in tried && sNeg > bestScore) { bestId = id; bestDirection = false; bestScore = sNeg }
            }
            // Every frontier candidate exhausted by `tried` → terminate descent.
            if (bestId < 0) break

            when (val result = session.pinBool(bestId, bestDirection)) {
                is PropagationResult.Implied -> {
                    decisionIds += bestId
                    decisionDirs += bestDirection
                    decisionScores += bestScore
                }
                is PropagationResult.Unsat -> {
                    // The failed pin stays in the session until we pop it.
                    session.popLast()
                    tried += bestId to bestDirection

                    // Conflict-directed backjump: find the most-recent OF OUR DECISIONS
                    // that's in the conflict set; pop the session back past it and mark
                    // its (id, dir) as tried so we'll pick a different direction or move
                    // on. If no decision is in the conflict, the contradiction is rooted
                    // in problem constraints alone — nothing to backtrack to.
                    val conflictingIdx = decisionIds.indices.reversed()
                        .firstOrNull { idx -> decisionIds[idx] in result.conflictBools }
                    if (conflictingIdx != null) {
                        val popVar = decisionIds[conflictingIdx]
                        val popDir = decisionDirs[conflictingIdx]
                        session.popUntilUnpinned(VarKind.Bool, popVar)
                        tried += popVar to popDir
                        val n = decisionIds.size
                        decisionIds.subList(conflictingIdx, n).clear()
                        decisionDirs.subList(conflictingIdx, n).clear()
                        decisionScores.subList(conflictingIdx, n).clear()
                    }
                }
            }
        }

        // Happy path: pin set is satisfiable; klause finds a witness.
        val raw = proposeSample(rng, session.currentAssumptions())
        if (raw != null) return BanditSample.dithered(raw, space, rng)

        // Fallback path: soft-satisfy via LinearObjective.
        return runFallback(rng, decisionIds, decisionDirs, decisionScores)
    }

    private fun runFallback(
        rng: Random,
        decisionIds: List<Int>,
        decisionDirs: List<Boolean>,
        decisionScores: List<Double>,
    ): BanditSample {
        val fallback = optimizeFallback ?: throw NoFeasibleSampleException(
            "RandomForestBandit produced over-constrained decisions " +
                "(${decisionIds.size} greedy pins) and no optimizeFallback was supplied.",
        )
        val objective = buildSoftObjective(decisionIds, decisionDirs, decisionScores)
        val raw = fallback(objective, Assumptions.None)
            ?: throw NoFeasibleSampleException(
                "RandomForestBandit fallback could not find any feasible sample.",
            )
        return BanditSample.dithered(raw, space, rng)
    }

    /**
     * Build a [LinearObjective] whose minimum is the feasible sample that best matches
     * the bandit's greedy decisions weighted by their scores. Klause minimises, so:
     *   - decision (id, value=true) with score s → boolWeights[id] -= s (lower cost when true)
     *   - decision (id, value=false) with score s → boolWeights[id] += s (lower cost when false)
     * The sign of `s` already reflects the maximise/minimise direction from policy scoring.
     */
    private fun buildSoftObjective(
        ids: List<Int>,
        dirs: List<Boolean>,
        scores: List<Double>,
    ): LinearObjective {
        val n = space.compiled.problem.numBoolVars
        val weights = DoubleArray(n)
        for (i in ids.indices) {
            val id = ids[i]
            val s = scores[i]
            if (dirs[i]) weights[id] -= s else weights[id] += s
        }
        return LinearObjective(boolWeights = weights)
    }

    private fun signed(m: Double): Double = if (maximize) m else -m

    override fun predict(sample: BanditSample): Double {
        val row = projection.encode(sample)
        var sum = 0.0
        for (tree in trees) sum += tree.predict(row)
        return sum / trees.size
    }

    override fun update(sample: BanditSample, reward: Double, weight: Double) {
        rewards?.update(reward, 0L, weight)
        if (testAbsError != null) testAbsError.update(abs(reward - predict(sample)), 0L, weight)
        train(sample, reward, weight)
        if (trainAbsError != null) trainAbsError.update(abs(reward - predict(sample)), 0L, weight)
    }

    override fun train(sample: BanditSample, reward: Double, weight: Double) {
        val row = projection.encode(sample)
        if (!bagging) {
            for (tree in trees) tree.update(row, reward, weight)
            return
        }
        for (tree in trees) {
            val k = poissonOne(baggingRng)
            if (k > 0) tree.update(row, reward, weight * k)
        }
    }

    /** Knuth's Poisson sampler at λ=1. Returns 0/1/2/... with mass e^-1 / k!. */
    private fun poissonOne(rng: Random): Int {
        val l = exp(-1.0)
        var k = 0
        var p = 1.0
        do {
            k++
            p *= rng.nextDouble()
        } while (p > l)
        return k - 1
    }

    // Slice 1: serialisable forest state is a follow-up.
    override fun importData(data: ForestData) {}
    override fun exportData(): ForestData = ForestData(trees.map { DecisionTreeData() })

    companion object {
        /**
         * Convenience constructor: builds [nbrTrees] trees over the [space]'s default
         * split-candidate pool, each with its own per-leaf mtry subspace (the
         * Breiman default `⌈√p⌉` unless overridden). Seeded deterministically from
         * [randomSeed] so the same call always produces the same forest.
         */
        fun <R : Result> build(
            space: CompiledDecisionSpace,
            policy: BanditPolicy<R>,
            proposeSample: (Random, Assumptions) -> Sample?,
            nbrTrees: Int = 10,
            mtry: Int? = null,
            config: TreeConfig = TreeConfig(),
            retryBudget: Int = 32,
            optimizeFallback: ((LinearObjective, Assumptions) -> Sample?)? = null,
            randomSeed: Int = System.currentTimeMillis().toInt(),
            maximize: Boolean = true,
            rewards: SeriesStat<*>? = null,
            trainAbsError: SeriesStat<*>? = null,
            testAbsError: SeriesStat<*>? = null,
        ): RandomForestBandit<R> {
            val candidates = defaultSplitCandidates(space)
            val k = (mtry ?: defaultMtry(candidates.size)).coerceAtMost(candidates.size)
            val perTreeConfig = config.copy(mtry = k)
            val seedRng = Random(randomSeed)
            val trees = (0 until nbrTrees).map {
                Tree(policy, candidates, perTreeConfig, randomSeed = seedRng.nextInt())
            }
            return RandomForestBandit(
                space = space,
                policy = policy,
                proposeSample = proposeSample,
                trees = trees,
                retryBudget = retryBudget,
                optimizeFallback = optimizeFallback,
                randomSeed = randomSeed,
                maximize = maximize,
                rewards = rewards,
                trainAbsError = trainAbsError,
                testAbsError = testAbsError,
            )
        }

        /** Classic RF heuristic: ⌈√p⌉ candidates per leaf, with floor 1. */
        private fun defaultMtry(p: Int): Int =
            if (p <= 0) 0 else kotlin.math.ceil(kotlin.math.sqrt(p.toDouble())).toInt().coerceAtLeast(1)
    }
}
