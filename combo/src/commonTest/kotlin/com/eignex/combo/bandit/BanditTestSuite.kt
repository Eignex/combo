package com.eignex.combo.bandit

import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.shared.Learner
import com.eignex.combo.shared.LearnerData
import com.eignex.combo.shared.PredictionLearner
import com.eignex.klause.solver.Sample
import com.eignex.kumulant.stat.summary.MeanStat
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Spec for building a bandit under test. Concrete suites override [build] with a
 * factory that produces a [Bandit] over the supplied decision space, candidate sample
 * pool, and bookkeeping knobs.
 *
 * The pool is provided so list-style bandits can use it directly; bandits that don't
 * consume the pool (e.g. linear bandits sampling via klause every round) can ignore it.
 *
 * Each subclass typically picks a [com.eignex.kumulant.bandit.univariate.BanditPolicy] / linear model
 * / etc. inside [build] and wires it.
 */
abstract class BanditTestSuite<D : LearnerData> {

    /** Build the bandit. Called fresh for every test. */
    abstract fun build(
        space: CompiledDecisionSpace,
        samples: List<Sample>,
        randomSeed: Int,
        maximize: Boolean,
        rewards: MeanStat? = null,
    ): Learner<D>

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

    private fun enumerateAll(): List<Sample> = (0..3).map { bits ->
        Sample(booleanArrayOf(bits and 1 != 0, bits and 2 != 0), longArrayOf())
    }

    // -- Contract tests ----------------------------------------------------------------

    @Test
    fun `chooseOrThrow should return a feasible sample`() {
        val (_, space) = freshSpace()
        val samples = enumerateAll()
        val bandit = build(space, samples, randomSeed = 1, maximize = true)

        repeat(3) {
            val pick = bandit.chooseOrThrow()
            // Feasibility for this space (no constraints) reduces to "sample dimensions match".
            assertEquals(space.compiled.problem.numBoolVars, pick.bools.size)
            assertEquals(space.compiled.problem.numIntVars, pick.ints.size)
        }
    }

    @Test
    fun `update should fold rewards into the optional sink`() {
        val (_, space) = freshSpace()
        val samples = enumerateAll()
        val sink = MeanStat()
        val bandit = build(space, samples, randomSeed = 2, maximize = true, rewards = sink)

        val rng = Random(2)
        repeat(4) {
            val pick = bandit.chooseOrThrow()
            val r = rng.nextDouble()
            bandit.update(pick, r)
        }
        val snap = sink.read()
        assertEquals(4.0, snap.totalWeights, 1e-9)
        assertTrue(snap.mean in 0.0..1.0)
    }

    @Test
    fun `bandit accepts rewards for each feasible arm`() {
        val (model, space) = freshSpace()
        val samples = enumerateAll()
        val sink = MeanStat()
        val bandit = build(space, samples, randomSeed = 3, maximize = true, rewards = sink)
        val aId = space.compiled.boolVarIdByName.getValue(model.a.name)
        for (sample in samples) {
            bandit.update(com.eignex.combo.decisions.BanditSample(sample), if (sample.bools[aId]) 1.0 else 0.0)
        }
        assertEquals(4.0, sink.read().totalWeights)
        assertTrue(bandit.optimalOrThrow().sample in samples)
    }

    @Test
    fun `optionalOrThrow should return some feasible sample after training`() {
        val (_, space) = freshSpace()
        val samples = enumerateAll()
        val bandit = build(space, samples, randomSeed = 4, maximize = true)
        // Exercise selection after a few distinct reward updates.
        val rng = Random(4)
        repeat(4) {
            val pick = bandit.chooseOrThrow()
            bandit.update(pick, rng.nextDouble())
        }
        val pick = bandit.optimalOrThrow()
        assertNotNull(pick)
        assertEquals(space.compiled.problem.numBoolVars, pick.bools.size)
    }

    @Test
    fun `choose vs chooseOrThrow agree when the context is feasible`() {
        val (_, space) = freshSpace()
        val samples = enumerateAll()
        val a = build(space, samples, randomSeed = 5, maximize = true)
        val b = build(space, samples, randomSeed = 5, maximize = true)

        // With the same seed, two bandits should walk the same path under no-context.
        val pickA = a.choose()
        val pickB = b.chooseOrThrow()
        assertNotNull(pickA)
        assertEquals(pickA, pickB)
    }
}

/**
 * Adds [PredictionLearner]-specific tests on top of [BanditTestSuite]: predict returns a
 * scalar, train accepts samples, train-then-predict reflects the trained reward.
 */
abstract class PredictionBanditTestSuite<D : LearnerData> : BanditTestSuite<D>() {

    abstract fun buildPrediction(
        space: CompiledDecisionSpace,
        samples: List<Sample>,
        randomSeed: Int,
        maximize: Boolean,
        rewards: MeanStat? = null,
    ): PredictionLearner<D>

    final override fun build(
        space: CompiledDecisionSpace,
        samples: List<Sample>,
        randomSeed: Int,
        maximize: Boolean,
        rewards: MeanStat?,
    ): Learner<D> = buildPrediction(space, samples, randomSeed, maximize, rewards)

    @Test
    fun `predict should produce a finite scalar`() {
        val (_, space) = freshSpace()
        val samples = (0..3).map { Sample(booleanArrayOf(it and 1 != 0, it and 2 != 0), longArrayOf()) }
        val bandit = buildPrediction(space, samples, randomSeed = 6, maximize = true)
        for (s in samples) {
            val y = bandit.predict(com.eignex.combo.decisions.BanditSample(s))
            assertTrue(y.isFinite(), "predict produced non-finite: $y")
        }
    }

    @Test
    fun `train should adjust predictions toward observed reward`() {
        val (model, space) = freshSpace()
        val samples = (0..3).map { Sample(booleanArrayOf(it and 1 != 0, it and 2 != 0), longArrayOf()) }
        val bandit = buildPrediction(space, samples, randomSeed = 7, maximize = true)

        // Pick a fixed sample, train it a few times with reward=1.0, verify predict rises.
        val target = com.eignex.combo.decisions.BanditSample(samples.first())
        val before = bandit.predict(target)
        repeat(3) { bandit.train(target, reward = 1.0) }
        val after = bandit.predict(target)
        assertTrue(
            after > before,
            "predict should grow after training on reward=1.0: before=$before, after=$after",
        )
    }
}
