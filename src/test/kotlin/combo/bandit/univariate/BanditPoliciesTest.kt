package combo.bandit.univariate

import combo.math.RunningVariance
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

abstract class BanditPolicyTest {
    abstract fun banditPolicy(): BanditPolicy

    @Test
    fun initialEvaluate() {
        val bp = banditPolicy()
        val data = bp.baseData()
        bp.addArm(data)
        bp.evaluate(data, 0L, true, Random)
    }

    @Test
    fun oneArmRound() {
        val bp = banditPolicy()
        val arm1 = bp.baseData()
        bp.addArm(arm1)
        bp.update(arm1, 0.2f, 10.0f)
        assertFalse(arm1.mean.isNaN())
        assertEquals(bp.baseData().nbrWeightedSamples + 10.0f, arm1.nbrWeightedSamples, 1E-6f)
    }

    @Test
    fun twoArmCompetitionMaximize() {
        val bp = banditPolicy()
        val arm1 = bp.baseData()
        val arm2 = bp.baseData()
        bp.addArm(arm1)
        bp.addArm(arm2)
        val rng = Random(0)
        for (i in 0 until 50) {
            val s1 = bp.evaluate(arm1, i.toLong(), true, rng)
            val s2 = bp.evaluate(arm2, i.toLong(), true, rng)
            if (s1 > s2) bp.update(arm1, 1.0f, 10.0f)
            else bp.update(arm2, 0.0f, 10.0f)
        }
        assertTrue(arm1.mean > arm2.mean)
        assertTrue(bp.evaluate(arm1, 50L, true, rng) > bp.evaluate(arm2, 50L, true, rng))
        assertEquals(bp.baseData().nbrWeightedSamples * 2 + 500.0f,
                arm1.nbrWeightedSamples + arm2.nbrWeightedSamples, 1E-6f)
    }

    @Test
    fun twoArmCompetitionMinimize() {
        val bp = banditPolicy()
        val arm1 = bp.baseData()
        val arm2 = bp.baseData()
        bp.addArm(arm1)
        bp.addArm(arm2)
        val rng = Random(0)
        for (i in 0 until 50) {
            val s1 = bp.evaluate(arm1, i.toLong(), false, rng)
            val s2 = bp.evaluate(arm2, i.toLong(), false, rng)
            if (s1 > s2) bp.update(arm1, 1.0f, 10.0f)
            else bp.update(arm2, 0.0f, 10.0f)
        }
        assertTrue(arm1.mean > arm2.mean)
        assertTrue(bp.evaluate(arm1, 50L, false, rng) < bp.evaluate(arm2, 50L, false, rng))
        assertEquals(bp.baseData().nbrWeightedSamples * 2 + 500.0f,
                arm1.nbrWeightedSamples + arm2.nbrWeightedSamples, 1E-6f)
    }
}

class ThompsonSamplingTest : BanditPolicyTest() {
    override fun banditPolicy() = ThompsonSampling(NormalPosterior)
}

class PooledThompsonSamplingTest : BanditPolicyTest() {
    override fun banditPolicy() = PooledThompsonSampling(
            HierarchicalNormalPosterior(PooledVarianceEstimator(RunningVariance(0.0f, 0.02f, 0.02f))))
}

class UCB1Test : BanditPolicyTest() {
    override fun banditPolicy() = UCB1()
}

class UCB1NormalTest : BanditPolicyTest() {
    override fun banditPolicy() = UCB1Normal()
}

class UCB1TunedTest : BanditPolicyTest() {
    override fun banditPolicy() = UCB1Tuned()
}

class EpsilonGreedyTest : BanditPolicyTest() {
    override fun banditPolicy() = EpsilonGreedy()
}

class EpsilonDecreasingTest : BanditPolicyTest() {
    override fun banditPolicy() = EpsilonDecreasing()
}
