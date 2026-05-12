package combo.bandit.univariate

import combo.math.DataSample
import combo.math.VarianceEstimator
import combo.math.VoidSample
import combo.util.RandomSequence
import java.util.concurrent.atomic.AtomicLong
import kotlin.jvm.JvmOverloads

/**
 * A univariate bandit with fixed number of arms. It stores one estimator per arm.
 */
class MultiArmedBandit @JvmOverloads constructor(
    val nbrArms: Int,
    val banditPolicy: BanditPolicy,
    override val randomSeed: Int = System.currentTimeMillis().toInt(),
    override val maximize: Boolean = true,
    override val rewards: DataSample = VoidSample) : UnivariateBandit<List<VarianceEstimator>> {

    init {
        require(nbrArms > 0)
    }

    private val randomSequence = RandomSequence(randomSeed)
    private val step = AtomicLong()
    private val data: Array<VarianceEstimator> = Array(nbrArms) { banditPolicy.baseData().also { banditPolicy.addArm(it) } }

    /**
     * Select the next bandit to use. Indexed from 0 to [nbrArms].
     */
    override fun choose(): Int {
        val t = step.getAndIncrement()
        val rng = randomSequence.next()
        return (0 until nbrArms).maxByOrNull { banditPolicy.evaluate(data[it], t, maximize, rng) }!!
    }

    override fun update(armIndex: Int, result: Float, weight: Float) {
        rewards.accept(result, weight)
        banditPolicy.update(data[armIndex], result, weight)
    }

    override fun updateAll(armIndices: IntArray, results: FloatArray, weights: FloatArray?) {
        require(armIndices.size == results.size) { "Arrays must be same length." }
        if (weights != null) require(weights.size == results.size) { "Arrays must be same length." }
        for (i in armIndices.indices) {
            val weight = weights?.get(i) ?: 1.0f
            val value = results[i]
            rewards.accept(value, weight)
            banditPolicy.update(data[armIndices[i]], value, weight)
        }
    }

    /**
     * The array must be the same length as [nbrArms].
     */
    override fun importData(data: List<VarianceEstimator>, replace: Boolean) {
        require(data.size == nbrArms) { "Inconsistent array length with number of arms." }
        if (replace) {
            for (i in 0 until nbrArms) {
                banditPolicy.removeArm(this.data[i])
                banditPolicy.addArm(data[i])
            }
        } else {
            for (i in 0 until nbrArms) {
                banditPolicy.removeArm(this.data[i])
                this.data[i] = this.data[i].combine(data[i])
                banditPolicy.addArm(this.data[i])
            }
        }
    }

    /**
     * Exports all data to use for external storage. They can be used in a new [UnivariateBandit] instance that
     * continues optimizing through the [importData] function. The order of the returned array must be maintained.
     */
    override fun exportData(): List<VarianceEstimator> {
        val list = ArrayList<VarianceEstimator>()
        for (i in data.indices)
            list.add(data[i].copy())
        return list
    }

    class Builder(val nbrArms: Int, val banditPolicy: BanditPolicy) {
        private var randomSeed: Int = System.currentTimeMillis().toInt()
        private var maximize: Boolean = true
        private var rewards: DataSample = VoidSample

        /** Set the random seed to a specific value to have a reproducible algorithm. By default current system time. */
        fun randomSeed(randomSeed: Int) = apply { this.randomSeed = randomSeed }

        /** Whether the bandit should maximize or minimize the total rewards. By default true. */
        fun maximize(maximize: Boolean) = apply { this.maximize = maximize }

        /** All rewards are added to this for inspecting how well the bandit performs. By default [combo.math.VoidSample]. */
        fun rewards(rewards: DataSample) = apply { this.rewards = rewards }

        /** Build bandit that can be used in parallel. */
        fun parallel() = ParallelUnivariateBandit.Builder(this)

        fun build() = MultiArmedBandit(nbrArms, banditPolicy, randomSeed, maximize, rewards)
    }
}
