package combo.sat.optimizers

import combo.sat.*
import combo.expressions.Conjunction
import combo.util.*

/**
 * This [Optimizer] uses brute force. It can only solve small and easy problems.
 * @param problem the problem contains the [combo.expressions.Constraint]s and the number of variables.
 * @param randomSeed Set the random seed to a specific value to have a reproducible algorithm.
 * @param timeout The solver will abort after timeout in milliseconds have been reached, without a real-time guarantee.
 * @param propagateAssumptions If true then perform unit propagation before solving when assumptions are used.
 * @param instanceFactory Determines the [Instance] that will be created for solving.
 * @param maxOptimizationInstances Maximum number of instances that will be observed when optimizing.
 *
 */
class ExhaustiveSolver(val problem: Problem, override val randomSeed: Int = System.currentTimeMillis().toInt(),
                       override val timeout: Long = -1L,
                       val propagateAssumptions: Boolean = true,
                       val instanceFactory: InstanceFactory = BitArrayFactory,
                       val maxOptimizationInstances: Int = 1000) : Optimizer<ObjectiveFunction> {

    private val randomSequence = RandomSequence(randomSeed)

    /**
     * The [guess] is used only if it satisfies all constraints.
     */
    override fun witnessOrThrow(assumptions: IntCollection, guess: Instance?): Instance {
        val propAssumptions = propAssumptions(assumptions)
        if (guess != null && (propAssumptions.isEmpty() || Conjunction(propAssumptions).satisfies(guess)) && problem.satisfies(guess))
            return guess
        val remap = createRemap(propAssumptions)
        val nbrVariables = problem.nbrValues - propAssumptions.size
        val end = if (timeout > 0) System.currentTimeMillis() + timeout else Long.MAX_VALUE
        return InstancePermutation(nbrVariables, instanceFactory, randomSequence.next())
                .asSequence()
                .takeWhile { System.currentTimeMillis() <= end }
                .map { remapInstance(propAssumptions, it, remap) }
                .filter { problem.satisfies(it) }
                .firstOrNull() ?: throw if (System.currentTimeMillis() > end) TimeoutException(timeout) else UnsatisfiableException()
    }

    override fun asSequence(assumptions: IntCollection): Sequence<Instance> {
        val propAssumptions = try {
            propAssumptions(assumptions)
        } catch (e: UnsatisfiableException) {
            return emptySequence()
        }
        val remap = createRemap(propAssumptions)
        val nbrVariables = problem.nbrValues - propAssumptions.size
        val end = if (timeout > 0) System.currentTimeMillis() + timeout else Long.MAX_VALUE
        return InstancePermutation(nbrVariables, instanceFactory, randomSequence.next())
                .asSequence()
                .takeWhile { System.currentTimeMillis() <= end }
                .map { remapInstance(propAssumptions, it, remap) }
                .filter { problem.satisfies(it) }
    }

    override fun optimizeOrThrow(function: ObjectiveFunction, assumptions: IntCollection, guess: Instance?): Instance {
        val opt = asSequence(assumptions).take(maxOptimizationInstances).minByOrNull {
            function.value(it)
        } ?: throw UnsatisfiableException()
        if (guess != null && problem.satisfies(guess)) {
            val f1 = function.value(guess)
            val f2 = function.value(opt)
            if (f1 < f2) return guess
        }
        return opt
    }

    private fun propAssumptions(assumptions: IntCollection): IntCollection {
        return if (propagateAssumptions && assumptions.isNotEmpty()) {
            val units = IntHashSet()
            units.addAll(assumptions)
            problem.unitPropagation(units)
            units
        } else {
            assumptions
        }
    }

    override val complete get() = true

    private fun createRemap(assumptions: IntCollection): IntArray {
        if (assumptions.isEmpty()) return EMPTY_INT_ARRAY
        val nbrVariables = problem.nbrValues - assumptions.size
        val themap = IntArray(nbrVariables)
        var ix = 0
        val taken = IntHashSet(assumptions.size * 2, nullValue = -1)
        assumptions.forEach { taken.add(it.toIx()) }
        for (i in 0 until nbrVariables) {
            while (taken.contains(ix)) ix++
            themap[i] = ix++
        }
        return themap
    }

    private fun remapInstance(assumptions: IntCollection, instance: Instance, remap: IntArray): Instance {
        return if (assumptions.isNotEmpty()) {
            val result = this.instanceFactory.create(problem.nbrValues)
            result.setAll(assumptions)
            for (i in instance.indices) {
                result[remap[i]] = instance.isSet(i)
            }
            result
        } else instance
    }
}
