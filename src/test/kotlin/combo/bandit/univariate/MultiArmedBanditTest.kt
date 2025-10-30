package combo.bandit.univariate

import combo.bandit.TestType
import combo.math.FullSample
import combo.math.nextBinomial
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class MultiArmedBanditTest {

    @Test
    fun zeroArms() {
        assertFailsWith(IllegalArgumentException::class) {
            MultiArmedBandit(0, UCB1())
        }
    }

    @Test
    fun minimizeVsMaximize() {
        val bandit1 = MultiArmedBandit(10, UCB1Tuned(), 1, true, FullSample())
        val bandit2 = MultiArmedBandit(10, UCB1Tuned(), 2, false, FullSample())

        val rng = Random(1L)
        for (i in 1..100) {
            val i1 = bandit1.choose()
            val i2 = bandit2.choose()
            val trials1 = (rng.nextInt(5) + 1)
            val trials2 = (rng.nextInt(5) + 1)
            val r1 = rng.nextBinomial(1f / (i1 + 1), trials1).toFloat()
            val r2 = rng.nextBinomial(1f / (i2 + 1), trials2).toFloat()
            bandit1.update(i1, r1 / trials1.toFloat(), trials1.toFloat())
            bandit2.update(i2, r2 / trials2.toFloat(), trials2.toFloat())
        }
        val sum1 = bandit1.rewards.values().sum()
        val sum2 = bandit2.rewards.values().sum()
        assertTrue(sum1 > sum2)
    }

    @Test
    fun randomSeedDeterministic() {
        val bandit1 = MultiArmedBandit(10, ThompsonSampling(NormalPosterior), randomSeed = 0)
        val bandit2 = MultiArmedBandit(10, ThompsonSampling(NormalPosterior), randomSeed = 0)
        val rng1 = Random(1L)
        val rng2 = Random(1L)
        val arms1 = generateSequence {
            bandit1.choose().also {
                bandit1.update(it, TestType.NORMAL.linearRewards(it.toFloat(), 1, rng1))
            }
        }.take(10).toList()
        val arm2 = generateSequence {
            bandit2.choose().also {
                bandit2.update(it, TestType.NORMAL.linearRewards(it.toFloat(), 1, rng2))
            }
        }.take(10).toList()
        for (i in 0 until 10) {
            assertEquals(arms1[i], arm2[i])
        }
        assertContentEquals(arms1, arm2)
    }

    @Test
    fun storeLoadStore() {
        val bandit = MultiArmedBandit(20, EpsilonDecreasing(), randomSeed = 1)
        for (i in 0 until 100) {
            val j = bandit.choose()
            bandit.update(j, TestType.BINOMIAL.linearRewards(j.toFloat() / 20, 1, Random))
        }
        val list1 = bandit.exportData()
        val bandit2 = MultiArmedBandit(20, EpsilonDecreasing(), randomSeed = 1)
        bandit2.importData(list1)

        assertEquals(bandit.choose(), bandit2.choose())
        assertEquals(list1.size, bandit2.exportData().size)
    }
}
