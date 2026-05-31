package combo.bandit.dt

import com.eignex.klause.solver.backtrack.BacktrackParams
import com.eignex.klause.solver.backtrack.BacktrackSolver
import com.eignex.klause.solver.localsearch.LocalSearchParams
import com.eignex.klause.solver.localsearch.LocalSearchSolver
import com.eignex.klause.solver.Sample
import com.eignex.klause.solver.SolveResult
import com.eignex.kumulant.stat.regression.tree.RegressionTreeConfig
import com.eignex.kumulant.stat.summary.MeanStat
import combo.bandit.PredictionLearner
import combo.bandit.PredictionBanditTestSuite
import com.eignex.kumulant.bandit.univariate.Greedy
import combo.decisions.CompiledDecisionSpace
import combo.decisions.DecisionSpace
import com.eignex.klause.ast.and
import com.eignex.klause.ast.not
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Plug [RandomForestBandit] into the shared harness. TinySpace has only two
 * features so per-leaf mtry doesn't usefully diversify the trees — half would be
 * blind to the rewarding feature, and bootstrap Thompson would pick those blind
 * trees 50% of the time. Pass `mtry = null` here so each tree sees both
 * candidates; convergence under the harness's strict win-rate threshold then
 * holds. Real mtry diversity wants a real-sized candidate pool.
 */
class RandomForestBanditTest : PredictionBanditTestSuite<ForestData>() {

    override fun buildPrediction(
        space: CompiledDecisionSpace,
        samples: List<Sample>,
        randomSeed: Int,
        maximize: Boolean,
        rewards: MeanStat?,
    ): PredictionLearner<ForestData> {
        val solver = LocalSearchSolver(space.compiled.problem)
        val session = com.eignex.klause.solver.localsearch.LocalSearchSession(solver)
        val backtrack = BacktrackSolver(space.compiled.problem)
        // Primary descent drives a dedicated session via push/pop assumption state; the
        // proposeSample/optimizeFallback cascade above remains the hard-problem fallback.
        val descentSession = KlauseDescentSession(
            com.eignex.klause.solver.localsearch.LocalSearchSession(solver),
        ) { rng -> LocalSearchParams(randomSeed = rng.nextLong()) }
        return RandomForestBandit.build(
            space = space,
            policy = Greedy(),
            descentSession = descentSession,
            // Cascade: LS first (fast common case); on null escalate to BacktrackSolver
            // (complete) so null from proposeSample means *definitive* UNSAT rather than
            // "LS gave up". Without the cascade, an LS budget miss silently routes the
            // bandit through the LinearObjective fallback and biases its training data.
            proposeSample = { rng, assumptions ->
                session.sample(LocalSearchParams(randomSeed = rng.nextLong(), assumptions = assumptions)).assignment
                    ?: backtrackSat(backtrack, rng, assumptions)
            },
            optimizeFallback = { objective, assumptions ->
                session.minimize(objective, LocalSearchParams(randomSeed = 0L, assumptions = assumptions)).assignment
                    ?: backtrack.minimize(objective, BacktrackParams(assumptions = assumptions)).assignment
            },
            nbrTrees = 6,
            mtry = null,
            config = RegressionTreeConfig(
                splitPeriod = 5,
                minSamplesSplit = 10.0,
                minSamplesLeaf = 2.0,
                delta = 0.1,
            ),
            retryBudget = 32,
            randomSeed = randomSeed,
            maximize = maximize,
            rewards = rewards,
        )
    }

    /**
     * With [Greedy] (deterministic) + `mtry = null` (every tree sees every candidate)
     * the only diversity source is online bagging. Disable it and the forest collapses
     * to N copies of the same tree; enable it and the per-tree leaf means diverge.
     * This sanity-checks that Poisson(1) reweighting actually does something.
     */
    @Test
    fun `bagging should diversify trees with deterministic policy`() {
        val (model, space) = freshSpace()
        val solver = LocalSearchSolver(space.compiled.problem)
        val sampler: (Random, com.eignex.klause.solver.Assumptions) -> Sample? =
            { rng, asmps -> solver.sample(LocalSearchParams(randomSeed = rng.nextLong(), assumptions = asmps)).assignment }

        fun forestWith(bagging: Boolean): RandomForestBandit {
            val cfg = RegressionTreeConfig(splitPeriod = 5, minSamplesSplit = 10.0, minSamplesLeaf = 2.0)
            val candidates = defaultSplitCandidates(space)
            val seedRng = Random(11)
            val trees = (0 until 8).map { Tree(Greedy(), candidates, cfg, randomSeed = seedRng.nextInt()) }
            return RandomForestBandit(
                space = space,
                policy = Greedy(),
                proposeSample = sampler,
                trees = trees,
                retryBudget = 32,
                bagging = bagging,
                randomSeed = 11,
                maximize = true,
                rewards = null,
            )
        }

        val baggedSpread = trainAndMeasureLeafSpread(model, forestWith(bagging = true))
        val noBagSpread = trainAndMeasureLeafSpread(model, forestWith(bagging = false))

        assertTrue(
            baggedSpread > noBagSpread,
            "bagging should produce more per-tree leaf-mean spread than no-bagging; got bagged=$baggedSpread, plain=$noBagSpread",
        )
    }

    private fun trainAndMeasureLeafSpread(model: TinySpace, forest: RandomForestBandit): Double {
        val aId = forest.space.compiled.boolVarIdByName[model.a.name]!!
        val rng = Random(42)
        // Bernoulli rewards depending on `a` only — gives trees the same signal to
        // converge on if they all saw the same data; bagging breaks that symmetry.
        repeat(200) {
            val s = forest.chooseOrThrow()
            val p = if (s.bools[aId]) 0.8 else 0.2
            forest.update(s, if (rng.nextDouble() < p) 1.0 else 0.0)
        }
        // Compute std-dev of per-tree predictions at the same fixed sample.
        val fixed = forest.optimalOrThrow()
        val perTree = forest.trees.map { it.predict(combo.bandit.dt.TreeFeatureProjection(forest.space).encode(fixed)) }
        val mean = perTree.average()
        val variance = perTree.map { (it - mean) * (it - mean) }.average()
        return kotlin.math.sqrt(variance)
    }

    /**
     * Sanity-check that greedy descent + klause propagation + LinearObjective fallback
     * stay correct under a non-trivial constraint structure: two bool decisions with an
     * exclusion constraint (`!(a && b)`). Greedy might pin both `a=true` and `b=true`,
     * which propagation should now catch as Unsat → the fallback then soft-satisfies
     * whichever decision had the higher score. Every choose call must still return a
     * feasible sample under the constraint.
     */
    @Test
    fun `propagation plus fallback should yield feasible samples under exclusion constraint`() {
        val schema = ConstrainedToggles()
        val space = schema.compileSpace()
        val solver = LocalSearchSolver(space.compiled.problem)
        val session = com.eignex.klause.solver.localsearch.LocalSearchSession(solver)
        val backtrack = BacktrackSolver(space.compiled.problem)
        val sampler: (Random, com.eignex.klause.solver.Assumptions) -> Sample? = { rng, asmps ->
            session.sample(LocalSearchParams(
                randomSeed = rng.nextLong(),
                maxFlips = 4_000L,
                assumptions = asmps,
            )).assignment ?: backtrackSat(backtrack, rng, asmps)
        }
        val bandit = RandomForestBandit.build(
            space = space,
            policy = Greedy(),
            proposeSample = sampler,
            optimizeFallback = { obj, asmps ->
                session.minimize(obj, LocalSearchParams(randomSeed = 0L, maxFlips = 4_000L, assumptions = asmps)).assignment
                    ?: backtrack.minimize(obj, BacktrackParams(assumptions = asmps)).assignment
            },
            nbrTrees = 4,
            mtry = null,
            config = RegressionTreeConfig(splitPeriod = 5, minSamplesSplit = 10.0, minSamplesLeaf = 2.0),
            randomSeed = 23,
            maximize = true,
        )
        val aId = space.compiled.boolVarIdByName.getValue("a")
        val bId = space.compiled.boolVarIdByName.getValue("b")
        val rng = Random(23)
        // Reward shape: a alone gives 1.0; b alone gives 0.6; constraint forbids both.
        repeat(400) {
            val s = bandit.chooseOrThrow()
            // Constraint is hard-enforced by klause; assert it.
            assertTrue(!(s.bools[aId] && s.bools[bId]),
                "constraint violated: a=${s.bools[aId]} b=${s.bools[bId]}")
            val reward = when {
                s.bools[aId] -> 1.0
                s.bools[bId] -> 0.6
                else -> 0.0
            } + (rng.nextDouble() - 0.5) * 0.1
            bandit.update(s, reward)
        }
        // Optimal under the converged forest should favour `a=true, b=false`.
        val best = bandit.optimalOrThrow()
        assertTrue(best.bools[aId], "optimal should pick a=true")
        assertTrue(!best.bools[bId], "optimal should pick b=false")
    }
}

/** Two bools with mutual exclusion — exercises klause's propagator inside RF descent. */
private class ConstrainedToggles : DecisionSpace() {
    val a by boolVar()
    val b by boolVar()
    val notBoth by constraint { !(a and b) }
}

/**
 * Definitive SAT/UNSAT arbiter via the complete [BacktrackSolver]. Returns the witness
 * on [SolveResult.Sat] (LS was wrong to give up); null otherwise (problem is genuinely
 * UNSAT under [assumptions] or the backtrack budget itself was exhausted, both of which
 * the bandit treats as "go to LinearObjective fallback").
 */
private fun backtrackSat(
    backtrack: BacktrackSolver,
    rng: Random,
    assumptions: com.eignex.klause.solver.Assumptions,
): Sample? = when (val r = backtrack.solve(BacktrackParams(randomSeed = rng.nextLong(), assumptions = assumptions))) {
    is SolveResult.Sat -> r.assignment
    else -> null
}
