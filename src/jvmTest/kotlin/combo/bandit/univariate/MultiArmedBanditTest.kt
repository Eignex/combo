package combo.bandit.univariate

import com.eignex.kumulant.bandit.univariate.BanditPolicy
import com.eignex.kumulant.bandit.univariate.BetaBernoulliTS
import com.eignex.kumulant.bandit.univariate.EpsilonGreedy
import com.eignex.kumulant.bandit.univariate.Greedy
import com.eignex.kumulant.bandit.univariate.MultiArmedBandit
import com.eignex.kumulant.bandit.univariate.NormalTS
import com.eignex.kumulant.bandit.univariate.UCB1
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MultiArmedBanditTest {

    private fun runBernoulli(
        policy: BanditPolicy<com.eignex.kumulant.stat.summary.BernoulliSumResult>,
        rounds: Int,
        seed: Int = 42,
    ): Int {
        val arms = doubleArrayOf(0.2, 0.5, 0.8)
        val bandit = MultiArmedBandit(arms.size, policy, Random(seed.toLong()))
        val rng = Random(seed.toLong())
        repeat(rounds) {
            val i = bandit.choose()
            val reward = if (rng.nextDouble() < arms[i]) 1.0 else 0.0
            bandit.update(i, reward)
        }
        // Best arm should dominate trials.
        return bandit.snapshot()
            .withIndex()
            .maxBy { (_, s) -> s.trials }
            .index
    }

    @Test
    fun `thompson sampling should converge to best bernoulli arm`() {
        val best = runBernoulli(BetaBernoulliTS(), rounds = 2000)
        assertEquals(2, best, "Thompson should favor arm 2 (p=0.8)")
    }

    @Test
    fun `ucb1 should converge to best bernoulli arm`() {
        val best = runBernoulli(UCB1(), rounds = 2000)
        assertEquals(2, best, "UCB1 should favor arm 2 (p=0.8)")
    }

    @Test
    fun `normal posterior should track means`() {
        val means = doubleArrayOf(-1.0, 0.0, 2.0)
        val policy = NormalTS()
        val bandit = MultiArmedBandit(means.size, policy, Random(1))
        val rng = Random(1)
        repeat(3000) {
            val i = bandit.choose()
            bandit.update(i, rng.nextDouble() * 2 - 1 + means[i])
        }
        val snap = bandit.snapshot()
        val bestArm = snap.withIndex().maxBy { (_, s) -> s.totalWeights }.index
        assertEquals(2, bestArm, "Normal Thompson should explore arm 2 most")
        // Estimated means should rank correctly.
        assertTrue(snap[2].mean > snap[1].mean)
        assertTrue(snap[1].mean > snap[0].mean)
    }

    @Test
    fun `epsilon greedy should converge`() {
        val arms = doubleArrayOf(0.1, 0.9)
        val bandit = MultiArmedBandit(arms.size, EpsilonGreedy(epsilon = 0.1), Random(7))
        val rng = Random(7)
        repeat(500) {
            val i = bandit.choose()
            bandit.update(i, if (rng.nextDouble() < arms[i]) 1.0 else 0.0)
        }
        assertEquals(1, bandit.snapshot().withIndex().maxBy { (_, s) -> s.totalWeights }.index)
    }

    @Test
    fun `greedy should run without error`() {
        // Greedy famously locks into the first apparent winner. We don't assert convergence —
        // just that the bandit runs and produces a well-formed snapshot.
        val arms = doubleArrayOf(0.1, 0.9)
        val bandit = MultiArmedBandit(arms.size, Greedy(), Random(7))
        val rng = Random(7)
        repeat(200) {
            val i = bandit.choose()
            bandit.update(i, if (rng.nextDouble() < arms[i]) 1.0 else 0.0)
        }
        val snap = bandit.snapshot()
        assertEquals(arms.size, snap.size)
        assertTrue(snap.any { it.totalWeights > 0 })
    }

    @Test
    fun `reward inversion should favor lowest mean`() {
        // kumulant's MultiArmedBandit no longer carries a `maximize` flag — it always
        // maximizes. Minimization is expressed by inverting the reward at the call site
        // (the same convention combo's own bandits apply via `signed`). Here we reward a
        // Bernoulli *failure*, so the lowest-p arm earns the most "successes" and wins.
        val arms = doubleArrayOf(0.2, 0.5, 0.8)
        val bandit = MultiArmedBandit(arms.size, BetaBernoulliTS(), Random(3))
        val rng = Random(3)
        repeat(1500) {
            val i = bandit.choose()
            bandit.update(i, if (rng.nextDouble() < arms[i]) 0.0 else 1.0)
        }
        val best = bandit.snapshot().withIndex().maxBy { (_, s) -> s.trials }.index
        assertEquals(0, best, "Inverted reward should make the lowest-p arm dominate")
    }
}
