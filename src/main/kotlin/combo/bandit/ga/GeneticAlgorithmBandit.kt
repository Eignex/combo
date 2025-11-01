package combo.bandit.ga

import combo.bandit.*
import combo.bandit.univariate.BanditPolicy
import combo.bandit.univariate.Greedy
import combo.ga.*
import combo.math.DataSample
import combo.math.VoidSample
import combo.math.nextGeometric
import combo.expressions.Constraint
import combo.expressions.Tautology
import combo.sat.*
import combo.expressions.Conjunction
import combo.sat.optimizers.LocalSearch
import combo.sat.optimizers.Optimizer
import combo.sat.optimizers.SatObjective
import combo.util.*
import java.util.concurrent.atomic.AtomicLong
import kotlin.collections.set
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random

/**
 * This bandit uses a univariate bandit algorithm, such as [combo.bandit.univariate.ThompsonSampling]. Each instance
 * in the candidate solutions (or gene pool) is a bandit. As new candidates are generated and poor ones are eliminated
 * the bandits arms are added or removed.
 *
 * @param problem The problem contains the [Constraint]s and the number of variables.
 * @param candidates Initialized instances in the gene pool.
 * @param banditPolicy The policy that the next bandit arm is chosen.
 * @param optimizer The optimizer will be used to generate [Instance]s that satisfy the constraints from the [Problem].
 * @param randomSeed Set the random seed to a specific value to have a reproducible algorithm.
 * @param maximize Whether the bandit should maximize or minimize the total rewards. By default true.
 * @param rewards All rewards are added to this for inspecting how well the bandit performs.
 * @param selection How candidates are selected to form the next candidate (see also [elimination]).
 * @param elimination How candidates are selected to eliminate the next candidate (see also [selection]).
 * @param eliminationPeriod Run the elimination with this period. If set to 1 it will run each update.
 * @param recombinationProbability Probability that the recombination operator is used when eliminating a candidate.
 * @param mutation Adds additional diversity to the candidate solutions (see also [mutationProbability])
 * @param mutationProbability Probability to apply [mutation] on new candidate.
 * @param allowDuplicates Whether duplicates are allowed in the candidates.
 * @param maxSolverRestarts Maximum number of restarts before solver gives up.
 * @param addAssumptions Whether candidates that are generated due to a no-match between candidates and assumptions should to the candidate solutions.
 */
class GeneticAlgorithmBandit(
        val problem: Problem,
        val candidates: BanditCandidates,
        val banditPolicy: BanditPolicy,
        val optimizer: Optimizer<SatObjective> = LocalSearch(problem),
        override val randomSeed: Int = System.currentTimeMillis().toInt(),
        override val maximize: Boolean = true,
        override val rewards: DataSample = VoidSample,
        val selection: SelectionOperator<BanditCandidates> = TournamentSelection(5),
        val elimination: SelectionOperator<BanditCandidates> = EliminationChain(SignificanceTestElimination(), TournamentElimination(3)),
        val eliminationPeriod: Int = 10,
        val recombinationProbability: Float = 0.5f,
        val mutation: MutationRate = FixedRateMutation(),
        val mutationProbability: Float = 0.1f,
        val allowDuplicates: Boolean = true,
        val maxSolverRestarts: Int = 10,
        val addAssumptions: Boolean = true)
    : Bandit<InstancesData> {

    init {
        for (e in candidates.estimators.values) banditPolicy.addArm(e)
    }

    private var randomSequence = RandomSequence(randomSeed)

    private var replacementCount = 0
    private val step = AtomicLong()

    private fun opt(opt: Boolean, assumptions: IntCollection, policy: BanditPolicy): Instance? {
        val assumption: Constraint = if (assumptions.isNotEmpty()) Conjunction(assumptions) else Tautology

        val t = if (opt) step.get() else step.getAndIncrement()
        val rng = randomSequence.next()
        val (instance, _) = candidates.estimators.maxByOrNull { (i, e) ->
            if (assumption.satisfies(i)) policy.evaluate(e, t, maximize, rng)
            else Float.NEGATIVE_INFINITY
        } ?: return null

        return if (!assumption.satisfies(instance)) {
            if (opt) null
            else {
                // This instance will always be a non-duplicate so we don't need to check for that
                val newInstance = optimizer.witnessOrThrow(assumptions)
                if (addAssumptions)
                    candidates.replaceCandidate(selectForElimination(true, rng), newInstance)
                newInstance
            }
        } else instance
    }

    companion object {

        fun newInstance(problem: Problem, optimizer: Optimizer<SatObjective>, rng: Random, recombinationProbability: Float, candidates: BanditCandidates,
                        selection: SelectionOperator<BanditCandidates>, mutationProbability: Float,
                        mutation: MutationRate): Instance {
            var newInstance: Instance? = null

            // Perform recombination
            val recombined = if (rng.nextFloat() < recombinationProbability) {
                val parent1 = selection.select(candidates, rng)
                val instance1 = candidates.instances[parent1]
                val parent2 = selection.select(candidates, rng)
                val instance2 = candidates.instances[parent2]
                val intersect = IntArrayList()
                for (i in instance1.indices) {
                    val lit1 = instance1.literal(i)
                    val lit2 = instance2.literal(i)
                    if (lit1 == lit2) intersect.add(lit1)
                }
                newInstance = optimizer.witness(intersect)
                parent1 != parent2
            } else false

            // Recombination failed/not performed?
            if (newInstance == null)
                newInstance = candidates.instances[selection.select(candidates, rng)].copy()

            // Perform mutation
            if (!recombined || rng.nextFloat() < mutationProbability) {
                val mr = mutation.rate(problem.nbrValues, rng)

                val mutatedInstance = newInstance.copy()
                var forcedAssumption: IntArrayList? = null
                var index = rng.nextGeometric(mr) - 1
                while (index < problem.nbrValues) {
                    if (forcedAssumption == null) {
                        forcedAssumption = IntArrayList()
                        forcedAssumption.add(!index.toLiteral(mutatedInstance.isSet(index)))
                    }
                    val literal = !index.toLiteral(mutatedInstance.isSet(index))
                    mutatedInstance.set(literal)
                    index += rng.nextGeometric(mr)
                }

                newInstance = optimizer.witness(forcedAssumption
                        ?: EmptyCollection, mutatedInstance)
                        ?: newInstance
            }
            return newInstance
        }
    }

    override fun chooseOrThrow(assumptions: IntCollection): Instance = opt(false, assumptions, banditPolicy)
            ?: throw UnsatisfiableException("No candidates matching assumptions found.")

    override fun optimalOrThrow(assumptions: IntCollection): Instance = opt(true, assumptions, Greedy)
            ?: throw UnsatisfiableException("No candidates matching assumptions found.")

    override fun update(instance: Instance, result: Float, weight: Float) {
        rewards.accept(result, weight)
        candidates.update(instance, result, weight)

        if (++replacementCount >= eliminationPeriod && candidates.step >= candidates.nbrCandidates) {

            val rng = randomSequence.next()
            // Select eliminated candidate
            val eliminated = selectForElimination(false, rng)
            if (eliminated < 0) return
            replacementCount = 0
            var newInstance: Instance = newInstance(problem, optimizer, rng, recombinationProbability, candidates, selection, mutationProbability, mutation)

            // Replace with random if it is a duplicate
            if (!allowDuplicates) {
                var k = 0
                while (!allowDuplicates && newInstance in candidates.estimators && k++ <= maxSolverRestarts)
                    newInstance = optimizer.witnessOrThrow()
                if (newInstance in candidates.estimators)
                    return
            }

            candidates.replaceCandidate(eliminated, newInstance)?.run {
                banditPolicy.removeArm(this)
            }
            if (!candidates.isDuplicated(newInstance))
                banditPolicy.addArm(candidates.estimators[newInstance]!!)
        }
    }

    override fun exportData(): InstancesData {
        val list = ArrayList<InstanceData>(candidates.nbrCandidates)
        for (i in 0 until candidates.nbrCandidates)
            list.add(InstanceData(candidates.instances[i], candidates.estimator(i)!!))
        return InstancesData(list)
    }

    override fun importData(data: InstancesData) {
        for ((i, newE) in data.instances) {
            val old = candidates.estimators[i] ?: continue
            banditPolicy.removeArm(old)
            val combined = old.combine(newE)
            candidates.estimators[i] = combined
            banditPolicy.addArm(combined)
        }
    }

    private fun selectForElimination(force: Boolean, rng: Random): Int {
        val e = elimination.select(candidates, rng)
        return if (force && e < 0) rng.nextInt(candidates.nbrCandidates)
        else e
    }

    class Builder(val problem: Problem, val banditPolicy: BanditPolicy) : BanditBuilder<InstancesData> {

        private var optimizer: Optimizer<SatObjective>? = null
        private var randomSeed: Int = System.currentTimeMillis().toInt()
        private var maximize: Boolean = true
        private var rewards: DataSample = VoidSample
        private var selection: SelectionOperator<BanditCandidates> = TournamentSelection(5)
        private var elimination: SelectionOperator<BanditCandidates> = EliminationChain(
                SignificanceTestElimination(),
                TournamentElimination(10))
        private var eliminationPeriod: Int = 10
        private var recombinationProbability: Float = 0.7f
        private var mutation: MutationRate = FixedRateMutation()
        private var mutationProbability: Float = 0.1f
        private var allowDuplicates: Boolean = true
        private var maxSolverRestarts: Int = 10
        private var addAssumptions: Boolean = true

        private var minEliminationSamples: Float = 4.0f
        private var candidateSize: Int? = null

        private var importedData: InstancesData? = null

        /** The optimizer will be used to generate [Instance]s that satisfy the constraints from the [Problem]. */
        fun optimizer(optimizer: Optimizer<SatObjective>) = apply { this.optimizer = optimizer }

        @Suppress("UNCHECKED_CAST")
        override fun suggestOptimizer(optimizer: Optimizer<*>) = optimizer(optimizer as Optimizer<SatObjective>)

        /** Set the random seed to a specific value to have a reproducible algorithm. */
        override fun randomSeed(randomSeed: Int) = apply { this.randomSeed = randomSeed }

        /** Whether the bandit should maximize or minimize the total rewards. By default true. */
        override fun maximize(maximize: Boolean) = apply { this.maximize = maximize }

        /** All rewards are added to this for inspecting how well the bandit performs. */
        override fun rewards(rewards: DataSample) = apply { this.rewards = rewards }

        /** The number of solution candidates that is generated by the search. This is the most important parameter to tweak. */
        fun candidateSize(candidateSize: Int) = apply { this.candidateSize = candidateSize }

        /** How candidates are selected to form the next candidate (see also [elimination]). */
        fun selection(selection: SelectionOperator<BanditCandidates>) = apply { this.selection = selection }

        /** How candidates are selected to eliminate the next candidate (see also [selection]). */
        fun elimination(elimination: SelectionOperator<BanditCandidates>) = apply { this.elimination = elimination }

        /** Run the elimination with this period. If set to 1 it will run each update. */
        fun eliminationPeriod(eliminationPeriod: Int) = apply { this.eliminationPeriod = eliminationPeriod }

        /** Probability that the recombination operator is used when eliminating a candidate. */
        fun recombinationProbability(recombinationProbability: Float) = apply { this.recombinationProbability = recombinationProbability }

        /** Adds additional diversity to the candidate solutions (see also [mutationProbability]) */
        fun mutation(mutation: MutationRate) = apply { this.mutation = mutation }

        /** Probability to apply [mutation] on new candidate. */
        fun mutationProbability(mutationProbability: Float) = apply { this.mutationProbability = mutationProbability }

        /** Whether duplicates are allowed in the candidates. */
        fun allowDuplicates(allowDuplicates: Boolean) = apply { this.allowDuplicates = allowDuplicates }

        /** Maximum number of restarts before solver gives up. */
        fun maxSolverRestarts(maxSolverRestarts: Int) = apply { this.maxSolverRestarts = maxSolverRestarts }

        /** Minimum number of sample that must be accrued before a candidate can be eliminated. */
        fun minEliminationSamples(minEliminationSamples: Float) = apply { this.minEliminationSamples = minEliminationSamples }

        /** addAssumptions Whether candidates that are generated due to a no-match between candidates and assumptions should to the candidate solutions. */
        fun addAssumptions(addAssumptions: Boolean) = apply { this.addAssumptions = addAssumptions }

        override fun importData(data: InstancesData) = apply { this.importedData = data }

        override fun parallel() = ParallelBandit.Builder(this).assumptionsLock(!addAssumptions)

        override fun build(): GeneticAlgorithmBandit {
            val optimizer = optimizer ?: LocalSearch.Builder(problem).randomSeed(randomSeed)
                    .cached().pNew(1.0f).maxSize(10).build()
            val candidates = if (importedData == null) {
                val candidateSize = candidateSize ?: max(10, min(problem.nbrValues * 2, 100))
                val instances: Array<Instance> = if (!allowDuplicates) {
                    if (optimizer.complete) optimizer.asSequence().take(candidateSize).toList().toTypedArray()
                    else optimizer.asSequence().take(candidateSize * 2).distinct().take(candidateSize).toList().toTypedArray()
                } else {
                    Array(candidateSize) {
                        var instance: Instance? = null
                        for (i in 0..maxSolverRestarts) {
                            instance = optimizer.witness() ?: continue
                            break
                        }
                        instance ?: throw IterationsReachedException(
                                "Max iterations $maxSolverRestarts reached during initialization.")
                    }
                }
                BanditCandidates(instances, minEliminationSamples, maximize, banditPolicy)
            } else {
                importedData!!
                val instances = importedData!!.instances.map { it.instance }.toTypedArray()
                val candidates = BanditCandidates(instances, minEliminationSamples, maximize, banditPolicy)
                for ((instance, data) in importedData!!) {
                    candidates.estimators[instance] = data.copy()
                }
                candidates.calculateMinMax()
                if (candidateSize != null && importedData!!.size < candidateSize!!) {
                    // Generate extra candidates
                    val rng = Random(randomSeed)
                    // TODO filter based on allowDuplicates
                    val extraInstances = Array(candidateSize!! - importedData!!.size) {
                        newInstance(problem, optimizer, rng, recombinationProbability, candidates, selection, mutationProbability, mutation)
                    }
                    val extraCandidates = BanditCandidates(instances + extraInstances, minEliminationSamples, maximize, banditPolicy)
                    for ((instance, data) in importedData!!)
                        extraCandidates.estimators[instance] = data.copy()
                    for (instance in extraInstances)
                        extraCandidates.estimators.getOrPut(instance) { banditPolicy.baseData() }
                    extraCandidates.calculateMinMax()
                    extraCandidates
                } else
                    candidates
            }
            return GeneticAlgorithmBandit(problem, candidates, banditPolicy, optimizer, randomSeed, maximize,
                    rewards, selection, elimination, eliminationPeriod, recombinationProbability,
                    mutation, mutationProbability, allowDuplicates, maxSolverRestarts, addAssumptions)
        }
    }
}
