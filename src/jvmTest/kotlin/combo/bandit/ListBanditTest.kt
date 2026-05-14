package combo.bandit

import com.eignex.klause.solver.LocalSearchParams
import com.eignex.klause.solver.LocalSearchSolver
import com.eignex.klause.solver.Sample
import com.eignex.kumulant.stat.summary.BernoulliSumResult
import com.eignex.kumulant.stat.summary.MeanStat
import combo.bandit.univariate.BinomialPosterior
import combo.bandit.univariate.ThompsonSampling
import combo.decisions.CompiledDecisionSpace
import combo.decisions.DecisionSpace
import combo.decisions.context
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

private class TinyBoolSpace : DecisionSpace() {
    val a by boolVar()
    val b by boolVar()
}

private class WithContext : DecisionSpace() {
    val a by boolVar()
    val premium by contextBool()
}

/** Suite contract: ListBandit over a klause-enumerated sample pool with Thompson + BinomialPosterior. */
class ListBanditSuiteTest : BanditTestSuite<ListBanditData>() {
    override fun build(
        space: CompiledDecisionSpace,
        samples: List<Sample>,
        randomSeed: Int,
        maximize: Boolean,
        rewards: MeanStat?,
    ): Bandit<ListBanditData> = ListBandit<BernoulliSumResult>(
        samples = samples,
        policy = ThompsonSampling(BinomialPosterior()),
        space = space,
        randomSeed = randomSeed,
        maximize = maximize,
        rewards = rewards,
    )
}

class ListBanditTest {

    @Test
    fun `choose should converge to the best Bernoulli arm`() {
        val space = TinyBoolSpace().compileSpace()
        val solver = LocalSearchSolver(space.compiled.problem)
        // 4 distinct samples = all (a, b) combinations.
        val samples = solver.enumerate(LocalSearchParams(maxFlips = 10_000L, randomSeed = 1L)).take(4).toList()
        assertEquals(4, samples.size)

        // Reward depends only on `a`: a=true → 0.8, a=false → 0.2.
        fun pSuccess(s: combo.decisions.BanditSample): Double = if (s.bools[0]) 0.8 else 0.2

        val bandit = ListBandit(
            samples = samples,
            policy = ThompsonSampling(BinomialPosterior()),
            space = space,
            randomSeed = 7,
        )
        val rng = Random(7)
        repeat(800) {
            val s = bandit.chooseOrThrow()
            val reward = if (rng.nextDouble() < pSuccess(s)) 1.0 else 0.0
            bandit.update(s, reward)
        }

        // Arm with a=true should have accumulated the most trials.
        val snap = bandit.snapshot()
        val bestIdx = snap.withIndex().maxBy { (_, r) -> r.trials }.index
        assertTrue(samples[bestIdx].bools[0], "winning arm should have a=true")
    }

    @Test
    fun `chooseOrThrow should respect context assumptions`() {
        val schema = WithContext()
        val space = schema.compileSpace()
        val solver = LocalSearchSolver(space.compiled.problem)
        // Enumerate (premium=true, a in {true,false}) and (premium=false, a in {true,false}).
        val samples = solver.enumerate(LocalSearchParams(maxFlips = 5_000L, randomSeed = 1L)).take(4).toList()

        val bandit = ListBandit(
            samples = samples,
            policy = ThompsonSampling(BinomialPosterior()),
            space = space,
            randomSeed = 11,
        )

        // chooseOrThrow under premium=true → returns a sample where premium=true.
        val premiumId = space.compiled.boolVarIdByName["premium"]!!
        val ctxT = context { set(schema.premium, true) }
        val ctxF = context { set(schema.premium, false) }
        repeat(20) {
            assertEquals(true, bandit.chooseOrThrow(ctxT).bools[premiumId])
            assertEquals(false, bandit.chooseOrThrow(ctxF).bools[premiumId])
        }
    }

    @Test
    fun `chooseOrThrow should throw when no sample matches context`() {
        val schema = WithContext()
        val space = schema.compileSpace()
        // Only one sample where premium=false.
        val premiumId = space.compiled.boolVarIdByName["premium"]!!
        val solver = LocalSearchSolver(space.compiled.problem)
        val onlyPremiumFalse = solver.enumerate(LocalSearchParams(maxFlips = 5_000L, randomSeed = 1L))
            .first { !it.bools[premiumId] }
        val bandit = ListBandit(
            samples = listOf(onlyPremiumFalse),
            policy = ThompsonSampling(BinomialPosterior()),
            space = space,
            randomSeed = 0,
        )
        val ctxTrue = context { set(schema.premium, true) }
        val err = runCatching { bandit.chooseOrThrow(ctxTrue) }.exceptionOrNull()
        assertNotNull(err)
        assertTrue(err is NoFeasibleSampleException)
    }
}
