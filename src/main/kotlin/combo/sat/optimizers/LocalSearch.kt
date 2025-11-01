package combo.sat.optimizers

import combo.math.permutation
import combo.expressions.Constraint
import combo.expressions.Tautology
import combo.sat.*
import combo.expressions.Conjunction
import combo.util.*
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * This solver implements local search for sat solving and optimization. For most easy optimization and solving tasks
 * this is the best option since there is very little overhead.
 *
 * Local search works by randomly generating a candidate solution. During each step a change in the [Instance] is taken
 * either randomly with probability [pRandomWalk] or a maximum greedy improvement is selected. The the candidate is
 * replaced by the improved candidate. If there are any unsatisfied constraints then the variables to inspect are taken
 * from the variables in the constraint, otherwise they are selected randomly. At most [maxConsideration] variables are
 * looked at for each step, if this parameter is set too low it can lead to premature termination of the algorithm.
 *
 * The specific algorithm used can be either simple Hill climbing, WalkSAT, WalkSAT with annealing, and/or Tabu search.
 * The type of algorithm is only decided through setting the various parameters,  [tabuListSize] (Tabu search),
 * [pRandomWalk] (WalkSAT), [pRandomWalkDecay] (Simulated annealing).
 *
 * @param problem the problem contains the [Constraint]s and the number of variables.
 * @param randomSeed Set the random seed to a specific value to have a reproducible algorithm.
 * @param timeout The solver will abort after timeout in milliseconds have been reached, without a real-time guarantee.
 * @param restarts The search will be restarted up to [restarts] number of time and the best value will be selected from each restart.
 * @param maxSteps Maximum number of steps for each of the [restarts].
 * @param pRandomWalk Chance of talking a random walk according to the WalkSAT algorithm.
 * @param pRandomWalkDecay Multiply the [pRandomWalk] by this during each step of the algorithm.
 * @param tabuListSize Keep a ring-buffer with blocked assignments during search. Size is always a power of 2.
 * @param instanceFactory Determines the [Instance] that will be created for solving.
 * @param initializer This determines how instances are given their starting values.
 * @param eps Threshold of improvement to stop current iteration in the search.
 * @param maxConsideration Maximum number of variables to consider during each search, set to [Int.MAX_VALUE] to disable.
 * @param propagateAssumptions Whether unit propagation before search is performed when assumptions are used.
 * @param transitiveImplications Used to propagate flips for implications, set to null for no propagations.
 */
class LocalSearch(val problem: Problem,
                  override val randomSeed: Int = System.currentTimeMillis().toInt(),
                  override val timeout: Long = -1L,
                  val restarts: Int = 5,
                  val maxSteps: Int = max(100, problem.nbrValues),
                  val pRandomWalk: Float = 0.1f,
                  val pRandomWalkDecay: Float = 0.95f,
                  tabuListSize: Int = Int.power2(min(problem.nbrValues, 2)),
                  val instanceFactory: InstanceFactory = BitArrayFactory,
                  val initializer: InstanceInitializer<*> = ConstraintCoercer(problem, WordRandomSet()),
                  val eps: Float = 1E-8f,
                  val maxConsideration: Int = max(20, min(100, problem.nbrValues / 5)),
                  val propagateAssumptions: Boolean = true,
                  val transitiveImplications: TransitiveImplications? = null) : Optimizer<ObjectiveFunction> {

    private val randomSequence = RandomSequence(randomSeed)

    val tabuListSize = if (tabuListSize == 0) 0 else Int.power2(tabuListSize)
    private val tabuMask = tabuListSize - 1

    override fun optimizeOrThrow(function: ObjectiveFunction, assumptions: IntCollection, guess: Instance?): Instance {
        val end = if (timeout > 0L) System.currentTimeMillis() + timeout else Long.MAX_VALUE

        val rng = randomSequence.next()
        val assumption: Constraint
        val p: Problem
        if (propagateAssumptions && assumptions.isNotEmpty()) {
            val units = IntHashSet()
            units.addAll(assumptions)
            p = Problem(problem.nbrValues, problem.unitPropagation(units, true))
            assumption = Conjunction(units)
        } else {
            p = problem
            assumption = if (assumptions.isEmpty()) Tautology else Conjunction(assumptions)
        }

        val adjustedMaxConsideration = max(2, min(maxConsideration, p.nbrValues))

        var bestValue = Float.POSITIVE_INFINITY
        var bestInstance: Instance? = null

        val lowerBound = function.lowerBound()
        val tabuBuffer = IntArray(tabuListSize) { -1 }
        var tabuI = 0

        for (restart in 1..restarts) {
            var pRandomWalk = pRandomWalk

            val instance: Instance
            if (guess != null && restart % 2 != 0) {
                instance = guess
            } else {
                instance = instanceFactory.create(p.nbrValues)
                @Suppress("UNCHECKED_CAST")
                (initializer as InstanceInitializer<ObjectiveFunction>).initialize(instance, assumption, rng, function)
            }
            val validator = Validator(p, instance, assumption)

            fun setReturnValue(value: Float) {
                if (value < bestValue && validator.totalUnsatisfied == 0) {
                    bestValue = value
                    bestInstance = validator.instance.copy()
                }
            }

            var prevValue = function.value(instance)
            if (prevValue.isNaN())
                throw NumericalInstabilityException("NaN function evaluation.")
            setReturnValue(prevValue)

            if (validator.totalUnsatisfied == 0 && (abs(bestValue - lowerBound) < eps || p.nbrValues == 0))
                return validator.instance

            for (step in 1..maxSteps) {
                val n: Int
                val ix: Int = if (pRandomWalk > rng.nextFloat()) {
                    if (validator.totalUnsatisfied > 0) validator.randomUnsatisfied(rng).literals.random(rng).toIx()
                    else rng.nextInt(p.nbrValues)
                } else {
                    val itr: IntIterator = if (validator.totalUnsatisfied > 0) {
                        val literals = validator.randomUnsatisfied(rng).literals
                        n = min(adjustedMaxConsideration, literals.size)
                        literals.permutation(rng)
                    } else {
                        n = min(adjustedMaxConsideration, p.nbrValues)
                        if (p.nbrValues > adjustedMaxConsideration)
                            OffsetIterator(1, permutation(p.nbrValues, rng).iterator())
                        else (1..p.nbrValues).iterator()
                    }
                    var maxSatImp = -validator.totalUnsatisfied
                    var maxOptImp = 0.0f
                    var bestIx = -1
                    for (k in 0 until n) {
                        val ix = itr.nextInt().toIx()
                        if (ix in tabuBuffer) continue
                        if (transitiveImplications != null && transitiveImplications.hasPropagations(!instance.literal(ix))) {
                            val copy = validator.copy()
                            //val copy = validator.instance.copy()
                            transitiveImplications.flipPropagate(copy, ix)
                            //val satScore = validator.totalUnsatisfied - p.violations(copy) //copy.totalUnsatisfied
                            val satScore = validator.totalUnsatisfied - copy.totalUnsatisfied
                            val optScore = prevValue - function.value(copy.instance)
                            if (satScore > maxSatImp || (satScore == maxSatImp && optScore > maxOptImp)) {
                                bestIx = ix
                                maxSatImp = satScore
                                maxOptImp = optScore
                            }
                        } else {
                            val satScore = validator.improvement(ix)
                            val optScore = function.improvement(instance, ix)
                            if (satScore > maxSatImp || (satScore == maxSatImp && optScore > maxOptImp)) {
                                bestIx = ix
                                maxSatImp = satScore
                                maxOptImp = optScore
                            }
                        }
                    }
                    bestIx
                }

                if (ix < 0)
                    break
                val improvement = if (transitiveImplications != null && transitiveImplications.hasPropagations(!instance.literal(ix))) {
                    transitiveImplications.flipPropagate(validator, ix)
                    val score = function.value(instance)
                    val imp = prevValue - score
                    prevValue = score
                    setReturnValue(score)
                    imp
                } else {
                    val imp = function.improvement(instance, ix)
                    val score = prevValue - imp
                    validator.flip(ix)
                    prevValue -= imp
                    setReturnValue(score)
                    imp
                }
                if (prevValue.isNaN())
                    throw NumericalInstabilityException("NaN function evaluation.")
                pRandomWalk *= pRandomWalkDecay
                if (tabuListSize > 0) {
                    tabuBuffer[tabuI] = ix
                    tabuI = (tabuI + 1) and tabuMask
                }
                if (validator.totalUnsatisfied == 0) {
                    if (abs(bestValue - lowerBound) < eps) return validator.instance
                    else if (improvement < eps) break
                }
                if (System.currentTimeMillis() > end) break
            }
            if (System.currentTimeMillis() > end) break
        }
        return bestInstance
                ?: (if (System.currentTimeMillis() > end) throw TimeoutException(timeout) else throw IterationsReachedException(restarts))
    }

    override fun witnessOrThrow(assumptions: IntCollection, guess: Instance?) = optimizeOrThrow(SatObjective, assumptions, guess)

    /**
     * Problem is the only mandatory parameter.
     */
    class Builder(val problem: Problem) :OptimizerBuilder<ObjectiveFunction> {
        private var randomSeed: Int = System.currentTimeMillis().toInt()
        private var timeout: Long = -1L
        private var restarts: Int = 5
        private var maxSteps: Int = max(100, problem.nbrValues)
        private var pRandomWalk: Float = 0.1f
        private var pRandomWalkDecay: Float = 0.95f
        private var tabuListSize: Int = Int.power2(min(problem.nbrValues, 2))
        private var instanceFactory: InstanceFactory = BitArrayFactory
        private var eps: Float = 1E-4f
        private var maxConsideration: Int = max(20, min(100, problem.nbrValues / 5))
        private var propagateAssumptions: Boolean = true

        private var propagateFlips: Boolean = true
        private var initializerType: InitializerType = InitializerType.PROPAGATE_COERCE
        private var initializerBias: Float = 0.5f
        private var initializerNoise: Float = 0.5f

        /** Set the random seed to a specific value to have a reproducible algorithm. */
        override fun randomSeed(randomSeed: Int) = apply { this.randomSeed = randomSeed }

        /** The solver will abort after timeout in milliseconds have been reached, without a real-time guarantee. */
        override fun timeout(timeout: Long) = apply { this.timeout = timeout }

        /** The search will be restarted up to [restarts] number of time and the best value will be selected from each restart. */
        fun restarts(restarts: Int) = apply { this.restarts = restarts }

        /** Maximum number of steps for each of the [restarts]. */
        fun maxSteps(maxSteps: Int) = apply { this.maxSteps = maxSteps }

        /** Chance of talking a random walk according to the WalkSAT algorithm. */
        fun pRandomWalk(pRandomWalk: Float) = apply { this.pRandomWalk = pRandomWalk }

        /** Multiply the [pRandomWalk] by this during each step of the algorithm. */
        fun pRandomWalkDecay(pRandomWalkDecay: Float) = apply { this.pRandomWalkDecay = pRandomWalkDecay }

        /** Keep a ring-buffer with blocked assignments during search. Size is always a power of 2. */
        fun tabuListSize(tabuListSize: Int) = apply { this.tabuListSize = tabuListSize }

        /** Whether to use sparse or dense bit array as instance. */
        fun sparse(sparse: Boolean) = apply { if (sparse) instanceFactory = SparseBitArrayFactory else BitArrayFactory }

        /** Type of initialization strategy. */
        fun initializer(initializer: InitializerType) = apply { this.initializerType = initializer }

        /** Preference when randomizing for initializing each value with 1 (bias close to 1) or 0 (bias close to 0). */
        fun initializerBias(initializerBias: Float) = apply { this.initializerBias = initializerBias }

        /** Noise added to weights for [InitializerType.WEIGHT_MAX] for [LinearObjective]. */
        fun initializerNoise(initializerNoise: Float) = apply { this.initializerNoise = initializerNoise }

        /** Threshold of improvement to stop current iteration in the search. */
        fun stallEps(eps: Float) = apply { this.eps = eps }

        /** Maximum number of variables to consider during each search, set to [Int.MAX_VALUE] to disable. */
        fun maxConsideration(maxConsideration: Int) = apply { this.maxConsideration = maxConsideration }

        /** Whether unit propagation before search is performed when assumptions are used. */
        fun propagateAssumptions(propagateAssumptions: Boolean) = apply { this.propagateAssumptions = propagateAssumptions }

        /** Whether flips propagate during search, each flip will be more expensive but more effective. */
        fun propagateFlips(propagateFlips: Boolean) = apply { this.propagateFlips = propagateFlips }

        /** Wrap this in a cached optimizer. */
        fun cached() = CachedOptimizer.Builder(build())

        fun fallbackCached() = cached().pNew(1f).maxSize(10)

        override fun build(): LocalSearch {

            val digraph = if (propagateFlips || initializerType == InitializerType.PROPAGATE_COERCE
                    || initializerType == InitializerType.WEIGHT_MAX_PROPAGATE_COERCE) TransitiveImplications(problem)
            else null

            val randomizer = if (initializerBias > 0.999f) RandomSet(initializerBias)
            else if (initializerBias == 0.0f) NoInitializer(false)
            else if (initializerBias <= 0.01f) GeometricRandomSet(initializerBias)
            else WordRandomSet(initializerBias)
            val init = when (initializerType) {
                InitializerType.WEIGHT_MAX -> WeightSet(initializerNoise)
                InitializerType.RANDOM -> randomizer
                InitializerType.COERCE -> ConstraintCoercer(problem, randomizer)
                InitializerType.PROPAGATE_COERCE -> ImplicationConstraintCoercer(problem, digraph!!, randomizer)
                InitializerType.WEIGHT_MAX_PROPAGATE_COERCE -> ImplicationConstraintCoercer(problem, digraph!!, WeightSet(initializerNoise))
                InitializerType.NONE -> NoInitializer(true)
            }

            return LocalSearch(problem, randomSeed = randomSeed, timeout = timeout, restarts = restarts,
                    maxSteps = maxSteps, pRandomWalk = pRandomWalk, pRandomWalkDecay = pRandomWalkDecay,
                    tabuListSize = tabuListSize, instanceFactory = instanceFactory, initializer = init, eps = eps,
                    maxConsideration = maxConsideration, transitiveImplications = if (propagateFlips) digraph else null,
                    propagateAssumptions = propagateAssumptions)
        }
    }
}
