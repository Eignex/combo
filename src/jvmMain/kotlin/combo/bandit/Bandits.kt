package combo.bandit

import combo.math.DataSample
import combo.math.VoidSample
import combo.sat.Instance
import combo.sat.Problem
import combo.sat.ValidationException
import combo.sat.optimizers.LocalSearch
import combo.sat.optimizers.Optimizer
import combo.sat.optimizers.SatObjective
import combo.util.EmptyCollection
import combo.util.IntCollection
import kotlin.math.abs

/**
 * A bandit optimizes an online binary decision problem. These bandits are multi-variate,
 * ie. there are multiple binary decision variables.
 *
 * TODO further bandit explanation, rewards optimization, arms, choose/update loop, iid, stochastic non-adversial.
 */
interface Bandit<D : BanditData> {

    /**
     * Generate the next instance to try out, throwing [ValidationException] on failure.
     * @param assumptions these are values that must be set by the returned instance, the format is in Dimacs.
     */
    fun chooseOrThrow(assumptions: IntCollection = EmptyCollection): Instance

    /**
     * Generate the next instance to try out, returning null on failure.
     * @param assumptions these are values that must be set by the returned instance, the format is in Dimacs.
     */
    fun choose(assumptions: IntCollection = EmptyCollection) =
            try {
                chooseOrThrow(assumptions)
            } catch (e: ValidationException) {
                null
            }

    /**
     * Calculate the perceived optimal instance, throwing [ValidationException] on failure.
     * @param assumptions these are values that must be set by the returned instance, the format is in Dimacs.
     */
    fun optimalOrThrow(assumptions: IntCollection = EmptyCollection): Instance

    /**
     * Calculate the perceived optimal instance, returning null on failure.
     * @param assumptions these are values that must be set by the returned instance, the format is in Dimacs.
     */
    fun optimal(assumptions: IntCollection = EmptyCollection) =
            try {
                optimalOrThrow(assumptions)
            } catch (e: ValidationException) {
                null
            }

    /**
     * Update the result of an instance.
     *
     * @param instance the assigned values used for the result.
     * @param result the reward obtained. If this constitutes multiple rewards then set weight appropriately and divide
     * by the [weight].
     * @param weight update strength. Can be used to signal importance of a result. The higher value the more the
     * algorithm is updated. If these are number of trials (or observations) in for example a binomial distributed
     * reward, then the result should be divided by weight before calling update (ie. the [result] should be mean).
     */
    fun update(instance: Instance, result: Float, weight: Float = 1.0f)

    /**
     * Update multiple results, all arrays must be same length.
     */
    fun updateAll(instances: Array<Instance>, results: FloatArray, weights: FloatArray? = null) {
        require(instances.size == results.size) { "Arrays must be same length." }
        if (weights != null) require(weights.size == results.size) { "Arrays must be same length." }
        for (i in instances.indices)
            update(instances[i], results[i], weights?.get(i) ?: 1.0f)
    }

    /**
     * Add historic [data] to the bandit, this can be used to store and re-start the bandit.
     * Existing data is combined with the imported data importing, in general off-policy data can only be used by
     * [PredictionBandit]. It is not recommended to do import of any data that the bandit has already seen, since that
     * will cause an underestimation of variance in the rewards.
     */
    fun importData(data: D)

    /**
     * Exports all data to use for external storage. They can be used in a new [Bandit] instance that
     * continues optimizing through the [importData] function.
     */
    fun exportData(): D

    /**
     * Set the random seed to a specific value to have a reproducible algorithm. By default current system time.
     */
    val randomSeed: Int

    /**
     * Whether the bandit should maximize or minimize the total rewards. By default true.
     */
    val maximize: Boolean

    /**
     * All rewards are added to this for inspecting how well the bandit performs. By default [combo.math.VoidSample].
     */
    val rewards: DataSample
}

interface BanditData {
    fun migrate(from: IntArray, to: IntArray): BanditData
}

/**
 * A [PredictionBandit] uses a machine learning model as part of the algorithm. This machine learning algorithm can also
 * be used to make predictions about an [Instance].
 */
interface PredictionBandit<D : BanditData> : Bandit<D> {

    /**
     * The total absolute error obtained on a prediction before update.
     */
    val trainAbsError: DataSample

    /**
     * The total absolute error obtained on a prediction after update.
     */
    val testAbsError: DataSample

    /**
     * Evaluate the machine learning model on an [instance].
     */
    fun predict(instance: Instance): Float

    /**
     * Update the model only without adding [rewards], [testAbsError] or [trainAbsError].
     * This is called as part of the [train] method.
     */
    fun train(instance: Instance, result: Float, weight: Float)

    fun trainAll(instances: Array<Instance>, results: FloatArray, weights: FloatArray?) {
        for (i in instances.indices)
            train(instances[i], results[i], weights?.get(i) ?: 1.0f)
    }

    /**
     * Register rewards, test error, training error, and perform [train] on instance.
     */
    override fun update(instance: Instance, result: Float, weight: Float) {
        rewards.accept(result, weight)
        if (testAbsError != VoidSample)
            testAbsError.accept(abs((result - predict(instance)) * weight))
        train(instance, result, weight)
        if (trainAbsError != VoidSample)
            trainAbsError.accept(abs((result - predict(instance)) * weight))
    }

    override fun updateAll(instances: Array<Instance>, results: FloatArray, weights: FloatArray?) {
        for (i in results.indices) {
            val w = weights?.get(i) ?: 1.0f
            rewards.accept(results[i], w)
            if (testAbsError != VoidSample)
                testAbsError.accept(abs((results[i] - predict(instances[i])) * w))
        }
        trainAll(instances, results, weights)
        for (i in results.indices)
            if (trainAbsError != VoidSample)
                trainAbsError.accept(abs((results[i] - predict(instances[i])) * (weights?.get(i) ?: 1.0f)))
    }
}

interface BanditBuilder<D : BanditData> {
    fun build(): Bandit<D>

    /**
     * Initialize with historic [data] to the bandit.
     */
    fun importData(data: D): BanditBuilder<D>

    /** All rewards are added to this for inspecting how well the bandit performs. By default [combo.math.VoidSample]. */
    fun rewards(rewards: DataSample): BanditBuilder<D>

    /** Whether the bandit should maximize or minimize the total rewards. By default true. */
    fun maximize(maximize: Boolean): BanditBuilder<D>

    /** Set the random seed to a specific value to have a reproducible algorithm. By default current system time. */
    fun randomSeed(randomSeed: Int): BanditBuilder<D>

    fun suggestOptimizer(optimizer: Optimizer<*>): BanditBuilder<D>

    /** Build bandit that can be used in parallel. */
    fun parallel(): ParallelBandit.Builder<D>
}

interface PredictionBanditBuilder<D : BanditData> : BanditBuilder<D> {
    override fun build(): PredictionBandit<D>
    override fun importData(data: D): PredictionBanditBuilder<D>
    override fun rewards(rewards: DataSample): PredictionBanditBuilder<D>
    override fun maximize(maximize: Boolean): PredictionBanditBuilder<D>
    override fun randomSeed(randomSeed: Int): PredictionBanditBuilder<D>
    override fun suggestOptimizer(optimizer: Optimizer<*>): PredictionBanditBuilder<D>
    override fun parallel(): ParallelPredictionBandit.Builder<D>

    /** The total absolute error obtained on a prediction before update. */
    fun trainAbsError(trainAbsError: DataSample): PredictionBanditBuilder<D>

    /** The total absolute error obtained on a prediction after update. */
    fun testAbsError(testAbsError: DataSample): PredictionBanditBuilder<D>
}

class RandomBandit(val optimizer: Optimizer<SatObjective>, override val rewards: DataSample) : Bandit<Nothing> {
    override fun chooseOrThrow(assumptions: IntCollection) = optimizer.witnessOrThrow(assumptions)
    override fun optimalOrThrow(assumptions: IntCollection) = optimizer.witnessOrThrow(assumptions)
    override fun update(instance: Instance, result: Float, weight: Float) {}
    override fun importData(data: Nothing) {}
    override fun exportData() = error("Cannot export.")
    override val maximize: Boolean get() = true
    override val randomSeed: Int get() = optimizer.randomSeed

    class Builder(val problem: Problem) : BanditBuilder<Nothing> {

        private var rewards: DataSample = VoidSample
        private var randomSeed: Int = System.currentTimeMillis().toInt()
        private var optimizer: Optimizer<SatObjective>? = null

        override fun rewards(rewards: DataSample) = apply { this.rewards = rewards }
        override fun randomSeed(randomSeed: Int) = apply { this.randomSeed = randomSeed }
        fun optimizer(optimizer: Optimizer<SatObjective>) = apply { this.optimizer = optimizer }
        @Suppress("UNCHECKED_CAST")
        override fun suggestOptimizer(optimizer: Optimizer<*>) = optimizer(optimizer as Optimizer<SatObjective>)

        override fun importData(data: Nothing) = this
        override fun maximize(maximize: Boolean) = this

        override fun parallel() = ParallelBandit.Builder(this)

        override fun build() = RandomBandit(optimizer
                ?: LocalSearch.Builder(problem).randomSeed(randomSeed).fallbackCached().build(), rewards)
    }
}
