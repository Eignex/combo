package combo.bandit.univariate

import com.eignex.kumulant.core.Result
import com.eignex.kumulant.core.SeriesStat
import combo.util.RandomSequence
import java.util.concurrent.atomic.AtomicLong

/**
 * Univariate bandit with a fixed number of independent arms. The [policy] owns each
 * arm's accumulator (a kumulant [SeriesStat]); on each [choose] the bandit reads a fresh
 * snapshot per arm and asks the policy to score them.
 */
class MultiArmedBandit<R : Result>(
    val nbrArms: Int,
    val policy: BanditPolicy<R>,
    override val randomSeed: Int = System.currentTimeMillis().toInt(),
    override val maximize: Boolean = true,
) : UnivariateBandit<R> {

    init { require(nbrArms > 0) { "nbrArms must be positive, got $nbrArms" } }

    private val randomSequence = RandomSequence(randomSeed)
    private val step = AtomicLong()
    private val arms: Array<SeriesStat<R>> = Array(nbrArms) {
        policy.createArm().also { policy.addArm(it.read(0L)) }
    }

    override fun choose(): Int {
        val t = step.getAndIncrement()
        val rng = randomSequence.next()
        var bestIdx = 0
        var bestScore = Double.NEGATIVE_INFINITY
        for (i in 0 until nbrArms) {
            val score = policy.evaluate(arms[i].read(0L), t, maximize, rng)
            if (score > bestScore) { bestScore = score; bestIdx = i }
        }
        return bestIdx
    }

    override fun update(armIndex: Int, value: Double, weight: Double) {
        policy.update(arms[armIndex], value, weight)
    }

    override fun snapshot(): List<R> = arms.map { it.read(0L) }
}
