package com.eignex.combo.bandit

import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.decisions.context
import com.eignex.combo.shared.Learner
import com.eignex.combo.shared.NoFeasibleSampleException
import com.eignex.klause.solver.Sample
import com.eignex.kumulant.bandit.univariate.BetaBernoulliTS
import com.eignex.kumulant.bandit.univariate.Greedy
import com.eignex.kumulant.stat.summary.BernoulliSumResult
import com.eignex.kumulant.stat.summary.MeanStat
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

/** Suite contract: ListBandit over a fixed sample pool with [BetaBernoulliTS]. */
class ListBanditSuiteTest : BanditTestSuite<ListBanditData>() {
    override fun build(
        space: CompiledDecisionSpace,
        samples: List<Sample>,
        randomSeed: Int,
        maximize: Boolean,
        rewards: MeanStat?,
    ): Learner<ListBanditData> = ListBandit<BernoulliSumResult>(
        samples = samples,
        policy = BetaBernoulliTS(),
        space = space,
        randomSeed = randomSeed,
        maximize = maximize,
        rewards = rewards,
    )
}

class ListBanditTest {

    @Test
    fun `recorded rewards select the better arm without a convergence run`() {
        val space = TinyBoolSpace().compileSpace()
        val samples = (0..3).map { Sample(booleanArrayOf(it and 1 != 0, it and 2 != 0), longArrayOf()) }
        val bandit = ListBandit(samples, Greedy(priorWeight = 0.0), space, randomSeed = 7)
        for (sample in samples) {
            bandit.update(com.eignex.combo.decisions.BanditSample(sample), if (sample.bools[0]) 1.0 else 0.0)
        }
        assertTrue(bandit.snapshot().all { it.totalWeights == 1.0 })
        assertTrue(bandit.optimalOrThrow().bools[0])
    }

    @Test
    fun `chooseOrThrow should respect context assumptions`() {
        val schema = WithContext()
        val space = schema.compileSpace()
        // Enumerate (premium=true, a in {true,false}) and (premium=false, a in {true,false}).
        val samples = (0..3).map { Sample(booleanArrayOf(it and 1 != 0, it and 2 != 0), longArrayOf()) }

        val bandit = ListBandit(
            samples = samples,
            policy = BetaBernoulliTS(),
            space = space,
            randomSeed = 11,
        )

        // chooseOrThrow under premium=true → returns a sample where premium=true.
        val premiumId = space.compiled.boolVarIdByName["premium"]!!
        val ctxT = context { set(schema.premium, true) }
        val ctxF = context { set(schema.premium, false) }
        repeat(2) {
            assertEquals(true, bandit.chooseOrThrow(ctxT).bools[premiumId])
            assertEquals(false, bandit.chooseOrThrow(ctxF).bools[premiumId])
        }
    }

    @Test
    fun `chooseOrThrow should throw when no sample matches context`() {
        val schema = WithContext()
        val space = schema.compileSpace()
        // Only one sample where premium=false.
        val onlyPremiumFalse = Sample(booleanArrayOf(false, false), longArrayOf())
        val bandit = ListBandit(
            samples = listOf(onlyPremiumFalse),
            policy = BetaBernoulliTS(),
            space = space,
            randomSeed = 0,
        )
        val ctxTrue = context { set(schema.premium, true) }
        val err = runCatching { bandit.chooseOrThrow(ctxTrue) }.exceptionOrNull()
        assertNotNull(err)
        assertTrue(err is NoFeasibleSampleException)
    }
}
