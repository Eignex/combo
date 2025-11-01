package combo.bandit

import combo.bandit.univariate.BanditPolicy
import combo.bandit.univariate.Greedy
import combo.math.DataSample
import combo.math.VarianceEstimator
import combo.math.VoidSample
import combo.expressions.Constraint
import combo.expressions.Tautology
import combo.sat.*
import combo.expressions.Conjunction
import combo.sat.optimizers.ExhaustiveSolver
import combo.sat.optimizers.LocalSearch
import combo.sat.optimizers.Optimizer
import combo.util.*
import java.util.concurrent.atomic.AtomicLong

/**
 * This bandit uses an independent univariate bandit policy for each of a pre-defined [Instance].
 * Use the sequence method in [combo.sat.optimizers.Optimizer] to generate the instances.
 *
 * @param instances all arms to use by the bandit.
 * @param banditPolicy the policy that the next bandit arm is selected with.
 * @param priors set specific priors or use to import historic data after restarts.
 * @param randomSeed Set the random seed to a specific value to have a reproducible algorithm.
 * @param maximize Whether the bandit should maximize or minimize the total rewards. By default true.
 * @param rewards All rewards are added to this for inspecting how well the bandit performs.
 */
class ListBandit(instances: Array<Instance>,
                 val banditPolicy: BanditPolicy,
                 priors: Map<Instance, VarianceEstimator> = emptyMap(),
                 override val randomSeed: Int = System.currentTimeMillis().toInt(),
                 override val maximize: Boolean = true,
                 override val rewards: DataSample = VoidSample) : Bandit<InstancesData> {

    private val instanceData = LinkedHashMap<Instance, VarianceEstimator>().apply {
        instances.associateTo(this) {
            val e = priors[it] ?: banditPolicy.baseData()
            banditPolicy.addArm(e)
            it to e
        }
    }

    private var randomSequence = RandomSequence(randomSeed)
    private val step = AtomicLong()

    override fun importData(data: InstancesData) {
        for ((instance, e) in data.instances) {
            val old = this.instanceData[instance] ?: continue
            banditPolicy.removeArm(old)
            val combined = old.combine(e)
            this.instanceData[instance] = combined
            banditPolicy.addArm(combined)
        }
    }

    override fun exportData(): InstancesData {
        val itr = instanceData.iterator()
        return InstancesData(List(instanceData.size) {
            val (l, t) = itr.next()
            InstanceData(l, t)
        })
    }

    override fun update(instance: Instance, result: Float, weight: Float) {
        rewards.accept(result, weight)
        val e = instanceData[instance]
        if (e != null) banditPolicy.update(e, result, weight)
    }

    private fun opt(assumptions: IntCollection, policy: BanditPolicy, t: Long): Instance {
        val con: Constraint = if (assumptions.isNotEmpty()) Conjunction(assumptions) else Tautology
        val rng = randomSequence.next()
        val instance = instanceData.maxByOrNull {
            if (con.satisfies(it.key)) {
                policy.evaluate(it.value, t, maximize, rng)
            } else Float.NEGATIVE_INFINITY
        }?.key
        if (instance == null || !con.satisfies(instance))
            throw UnsatisfiableException("No instance matching assumption literals.")
        return instance
    }

    override fun optimalOrThrow(assumptions: IntCollection) = opt(assumptions, Greedy, step.get())
    override fun chooseOrThrow(assumptions: IntCollection) = opt(assumptions, banditPolicy, step.getAndIncrement())

    class Builder private constructor(
            val banditPolicy: BanditPolicy,
            private val problem: Problem? = null,
            private val limit: Int = 500,
            private val instances: MutableList<Instance>? = null)
        : BanditBuilder<InstancesData> {

        private var randomSeed: Int = System.currentTimeMillis().toInt()
        private var maximize: Boolean = true
        private var rewards: DataSample = VoidSample
        private var import: InstancesData? = null

        /**
         * Use pre-generated [instances] as base.
         */
        constructor(instances: Collection<Instance>, banditPolicy: BanditPolicy)
                : this(banditPolicy, instances = instances.toMutableList())

        /**
         * Generates up to [limit] instances using the specified constraints in the [problem].
         */
        constructor(problem: Problem, banditPolicy: BanditPolicy, limit: Int = 500) : this(
                banditPolicy, problem, limit)

        override fun importData(data: InstancesData) = apply { this.import = data }
        override fun randomSeed(randomSeed: Int) = apply { this.randomSeed = randomSeed }
        override fun maximize(maximize: Boolean) = apply { this.maximize = maximize }
        override fun rewards(rewards: DataSample) = apply { this.rewards = rewards }
        override fun parallel() = ParallelBandit.Builder(this)

        override fun suggestOptimizer(optimizer: Optimizer<*>) = this
        override fun build(): ListBandit {
            val priors = if (import != null) {
                val priors = HashMap<Instance, VarianceEstimator>()
                for ((instance, e) in import!!.instances)
                    priors[instance] = e.copy()
                priors
            } else emptyMap<Instance, VarianceEstimator>()
            val instances = if (instances != null) {
                if (import != null) {
                    val instanceSet = instances.toHashSet()
                    instanceSet.addAll(priors.keys)
                    instanceSet.toTypedArray()
                } else instances.toTypedArray()
            } else {
                problem!!
                val optimizer = if (problem.nbrValues <= 20) ExhaustiveSolver(problem, randomSeed)
                else LocalSearch.Builder(problem).restarts(Int.MAX_VALUE).randomSeed(randomSeed).build()
                val data = if (optimizer.complete) optimizer.asSequence().take(limit)
                else optimizer.asSequence().take(limit * 2).distinct().take(limit)
                if (import != null) {
                    (data + import!!.instances.asSequence().map { it.instance }).toSet().toTypedArray()
                } else data.toList().toTypedArray()
            }
            return ListBandit(instances = instances, banditPolicy = banditPolicy,
                    randomSeed = randomSeed, maximize = maximize, rewards = rewards, priors = priors)
        }
    }
}
