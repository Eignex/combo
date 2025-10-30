package combo.bandit.univariate

import combo.bandit.ParallelMode
import combo.math.BucketSample
import org.junit.Test
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.random.Random
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ParallelUnivariateBanditTest {

    @Test
    fun dataShouldNotBeDuplicated() {
        for (mode in ParallelMode.values()) {
            for (copies in 1..5) {
                val bandit = MultiArmedBandit.Builder(10, ThompsonSampling(NormalPosterior))
                        .rewards(BucketSample())
                        .parallel()
                        .copies(copies)
                        .mode(mode)
                        .batchSize(1..20)
                        .build()

                assertEquals(0, bandit.processUpdates(false))

                val rng = Random
                for (i in 0 until 10) {
                    bandit.update(rng.nextInt(10), rng.nextFloat())
                }
                assertEquals(0L, bandit.rewards.nbrSamples)
                assertEquals(0, bandit.exportData().sumBy { it.nbrSamples.toInt() })

                assertEquals(10, bandit.processUpdates(true))

                assertEquals(10, bandit.exportData().sumBy { it.nbrSamples.toInt() })
                assertEquals(10, bandit.exportData().sumBy { it.nbrSamples.toInt() })

                for (i in 0 until copies) {
                    assertEquals(10L, bandit.bandits[i].rewards.nbrSamples)
                    assertEquals(10, bandit.bandits[i].exportData().sumBy { it.nbrSamples.toInt() })
                }
            }
        }
    }

    @Test
    fun blockingModeWithinBatchSize() {
        val bandit = MultiArmedBandit.Builder(2, ThompsonSampling(BinomialPosterior))
                .rewards(BucketSample())
                .parallel()
                .copies(2)
                .mode(ParallelMode.BLOCKING)
                .batchSize(15..15)
                .build()

        val cdl1 = CountDownLatch(1)
        val cdl2 = CountDownLatch(1)
        val ok = AtomicBoolean(true)
        Thread {
            ok.set(ok.get() && bandit.processUpdates(false) == 0)
            cdl1.countDown()
            ok.set(ok.get() && bandit.processUpdates(true) == 15)
            ok.set(ok.get() && bandit.processUpdates(true) == 15)
            cdl2.countDown()
        }.start()
        cdl1.await()
        val rng = Random

        Thread {
            for (i in 0 until 30)
                bandit.update(rng.nextInt(2), rng.nextInt(2).toFloat())
        }.start()
        cdl2.await()
        assertTrue(ok.get())
    }

    @Test
    fun lockingModeGrowsBiggerThanBatchSize() {

        val bandit = MultiArmedBandit.Builder(10, ThompsonSampling(NormalPosterior))
                .rewards(BucketSample())
                .parallel()
                .copies(2)
                .mode(ParallelMode.LOCKING)
                .batchSize(1..20)
                .build()

        val rng = Random
        for (i in 0 until 1000)
            bandit.update(rng.nextInt(10), rng.nextFloat())

        do {
            val n = bandit.processUpdates(false)
            assertTrue(n <= 20)
        } while (n > 0)
        assertEquals(1000, bandit.exportData().sumBy { it.nbrSamples.toInt() })
    }

    @Test
    fun nonBlockingDoesNotBlock() {
        val bandit = MultiArmedBandit.Builder(10, ThompsonSampling(NormalPosterior))
                .rewards(BucketSample())
                .parallel()
                .copies(2)
                .mode(ParallelMode.NON_LOCKING)
                .batchSize(2..3)
                .build()
        for (i in 0 until 100)
            bandit.update(Random.nextInt(10), Random.nextFloat())
    }
}
