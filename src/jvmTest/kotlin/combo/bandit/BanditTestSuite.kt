package combo.bandit

import com.eignex.klause.solver.localsearch.LocalSearchParams
import com.eignex.klause.solver.localsearch.LocalSearchSolver
import com.eignex.klause.solver.Sample
import com.eignex.kumulant.stat.summary.MeanStat
import combo.decisions.CompiledDecisionSpace
import combo.decisions.Context
import combo.decisions.DecisionSpace
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import org.junit.Rule
import org.junit.rules.Timeout

/**
 * Spec for building a bandit under test. Concrete suites override [build] with a
 * factory that produces a [Bandit] over the supplied decision space, candidate sample
 * pool, and bookkeeping knobs.
 *
 * The pool is provided so list-style bandits can use it directly; bandits that don't
 * consume the pool (e.g. linear bandits sampling via klause every round) can ignore it.
 *
 * Each subclass typically picks a [combo.bandit.univariate.BanditPolicy] / linear model
 * / etc. inside [build] and wires it.
 */
abstract class BanditTestSuite<D : BanditData> {

    /** Per-test ceiling. Bandit loops that genuinely hang should fail fast, not stall CI. */
    @get:Rule
    val timeout: Timeout = Timeout.seconds(30)

    /** Build the bandit. Called fresh for every test. */
    abstract fun build(
        space: CompiledDecisionSpace,
        samples: List<Sample>,
        randomSeed: Int,
        maximize: Boolean,
        rewards: MeanStat? = null,
    ): Bandit<D>

    /**
     * Whether the bandit is expected to converge to the best arm with Bernoulli rewards
     * after [BERN_ROUNDS] rounds. Set to false for non-learning bandits like
     * [RandomBandit] — the convergence test will then merely sanity-check the loop.
     */
    open val converges: Boolean = true

    // -- Shared decision space used by the harness -------------------------------------

    /** Two bools — `a` and `b` — and no constraints. Yields 4 feasible samples. */
    protected class TinySpace : DecisionSpace() {
        val a by boolVar()
        val b by boolVar()
    }

    protected fun freshSpace(): Pair<TinySpace, CompiledDecisionSpace> {
        val model = TinySpace()
        return model to model.compileSpace()
    }

    private fun enumerateAll(space: CompiledDecisionSpace, seed: Long): List<Sample> {
        val solver = LocalSearchSolver(space.compiled.problem)
        return solver.enumerate(LocalSearchParams(maxFlips = 10_000L, randomSeed = seed))
            .take(4)
            .toList()
    }

    // -- Contract tests ----------------------------------------------------------------

    @Test
    fun `chooseOrThrow should return a feasible sample`() {
        val (_, space) = freshSpace()
        val samples = enumerateAll(space, seed = 1L)
        val bandit = build(space, samples, randomSeed = 1, maximize = true)

        repeat(20) {
            val pick = bandit.chooseOrThrow()
            // Feasibility for this space (no constraints) reduces to "sample dimensions match".
            assertEquals(space.compiled.problem.numBoolVars, pick.bools.size)
            assertEquals(space.compiled.problem.numIntVars, pick.ints.size)
        }
    }

    @Test
    fun `update should fold rewards into the optional sink`() {
        val (_, space) = freshSpace()
        val samples = enumerateAll(space, seed = 2L)
        val sink = MeanStat()
        val bandit = build(space, samples, randomSeed = 2, maximize = true, rewards = sink)

        val rng = Random(2)
        repeat(50) {
            val pick = bandit.chooseOrThrow()
            val r = rng.nextDouble()
            bandit.update(pick, r)
        }
        val snap = sink.read()
        assertEquals(50.0, snap.totalWeights, 1e-9)
        assertTrue(snap.mean in 0.0..1.0)
    }

    @Test
    fun `bandit should prefer higher-reward arms with Bernoulli payoff`() {
        val (model, space) = freshSpace()
        val samples = enumerateAll(space, seed = 3L)
        val bandit = build(space, samples, randomSeed = 3, maximize = true)

        // Reward depends on `a`: a=true gives p=0.8, a=false gives p=0.2.
        val aId = space.compiled.boolVarIdByName[model.a.name]!!
        val rng = Random(3)
        var nWins = 0
        var nTotal = 0
        repeat(BERN_ROUNDS) { round ->
            val s = bandit.chooseOrThrow()
            val p = if (s.bools[aId]) 0.8 else 0.2
            val r = if (rng.nextDouble() < p) 1.0 else 0.0
            bandit.update(s, r)
            // Score the *late* rounds — that's where convergence should show.
            if (round >= BERN_ROUNDS / 2) {
                nTotal++
                if (s.bools[aId]) nWins++
            }
        }
        if (converges) {
            assertTrue(nWins.toDouble() / nTotal > 0.7,
                "expected late-stage win rate > 0.7, got ${nWins}/${nTotal}")
        }
    }

    @Test
    fun `optionalOrThrow should return some feasible sample after training`() {
        val (_, space) = freshSpace()
        val samples = enumerateAll(space, seed = 4L)
        val bandit = build(space, samples, randomSeed = 4, maximize = true)
        // Varied rewards keep the bandit's internal model from collapsing to all-zeros
        // — for LinearBandit + LocalSearchSolver an all-zero objective triggers a
        // termination-bug in klause's minimize (restart loop ignores maxFlips).
        val rng = Random(4)
        repeat(50) {
            val pick = bandit.chooseOrThrow()
            bandit.update(pick, rng.nextDouble())
        }
        val pick = bandit.optimalOrThrow()
        assertNotNull(pick)
        assertEquals(space.compiled.problem.numBoolVars, pick.bools.size)
    }

    @Test
    fun `choose vs chooseOrThrow agree on the same context (no infeasibility here)`() {
        val (_, space) = freshSpace()
        val samples = enumerateAll(space, seed = 5L)
        val a = build(space, samples, randomSeed = 5, maximize = true)
        val b = build(space, samples, randomSeed = 5, maximize = true)

        // With the same seed, two bandits should walk the same path under no-context.
        val pickA = a.choose()
        val pickB = b.chooseOrThrow()
        assertNotNull(pickA)
        assertEquals(pickA, pickB)
    }

    companion object {
        const val BERN_ROUNDS: Int = 600
    }
}

/**
 * Adds [PredictionBandit]-specific tests on top of [BanditTestSuite]: predict returns a
 * scalar, train accepts samples, train-then-predict reflects the trained reward.
 */
abstract class PredictionBanditTestSuite<D : BanditData> : BanditTestSuite<D>() {

    abstract fun buildPrediction(
        space: CompiledDecisionSpace,
        samples: List<Sample>,
        randomSeed: Int,
        maximize: Boolean,
        rewards: MeanStat? = null,
    ): PredictionBandit<D>

    final override fun build(
        space: CompiledDecisionSpace,
        samples: List<Sample>,
        randomSeed: Int,
        maximize: Boolean,
        rewards: MeanStat?,
    ): Bandit<D> = buildPrediction(space, samples, randomSeed, maximize, rewards)

    @Test
    fun `predict should produce a finite scalar`() {
        val (_, space) = freshSpace()
        val solver = LocalSearchSolver(space.compiled.problem)
        val samples = solver.enumerate(LocalSearchParams(randomSeed = 6L)).take(4).toList()
        val bandit = buildPrediction(space, samples, randomSeed = 6, maximize = true)
        for (s in samples) {
            val y = bandit.predict(combo.decisions.BanditSample.undithered(s))
            assertTrue(y.isFinite(), "predict produced non-finite: $y")
        }
    }

    @Test
    fun `train should adjust predictions toward observed reward`() {
        val (model, space) = freshSpace()
        val solver = LocalSearchSolver(space.compiled.problem)
        val samples = solver.enumerate(LocalSearchParams(randomSeed = 7L)).take(4).toList()
        val bandit = buildPrediction(space, samples, randomSeed = 7, maximize = true)

        // Pick a fixed sample, train it many times with reward=1.0, verify predict rises.
        val target = combo.decisions.BanditSample.undithered(samples.first())
        val before = bandit.predict(target)
        repeat(100) { bandit.train(target, reward = 1.0) }
        val after = bandit.predict(target)
        assertTrue(after > before,
            "predict should grow after training on reward=1.0: before=$before, after=$after")
    }
}
