package combo.sat.optimizers

import combo.sat.Instance
import combo.sat.ValidationException
import combo.util.EmptyCollection
import combo.util.IntCollection

/**
 * An optimizer minimizes an [ObjectiveFunction]. It can also generate a random [witness] that satisfy the constraints
 * and iterate over the possible solutions with [asSequence].
 */
interface Optimizer<in O : ObjectiveFunction> : Iterable<Instance> {

    /**
     * Generates a random solution, ie. a witness.
     * @param assumptions these variables will be fixed during solving, in dimacs format.
     * @param guess starting point for search if one is provided. This instance will be reused if applicable.
     */
    fun witness(assumptions: IntCollection = EmptyCollection, guess: Instance? = null): Instance? {
        return try {
            witnessOrThrow(assumptions, guess)
        } catch (_: ValidationException) {
            null
        }
    }

    /**
     * @param assumptions these variables will be fixed during solving, in dimacs format.
     * @param guess starting point for search if one is provided. This instance will be reused if applicable.
     * @throws ValidationException if there is a logical error in the problem or a solution cannot be found with the
     * allotted resources.
     */
    fun witnessOrThrow(assumptions: IntCollection = EmptyCollection, guess: Instance? = null): Instance

    /**
     * Minimize the [function], optionally with the additional constraints in [assumptions].
     * Returns null if no instance can be found.
     * @param function the objective function to optimize on.
     * @param assumptions these variables will be fixed during solving, in dimacs format.
     * @param guess starting point for search if one is provided. This instance will be reused if applicable.
     */
    fun optimize(function: O, assumptions: IntCollection = EmptyCollection, guess: Instance? = null): Instance? {
        return try {
            optimizeOrThrow(function, assumptions, guess)
        } catch (e: ValidationException) {
            null
        }
    }

    /**
     * Minimize the [function], optionally with the additional constraints in [assumptions].
     * @param function the objective function to optimize on.
     * @param assumptions these variables will be fixed during solving, in dimacs format.
     * @param guess starting point for search if one is provided. This instance will be reused if applicable.
     * @throws ValidationException if there is a logical error in the problem or a solution cannot be found with the
     * allotted resources.
     */
    fun optimizeOrThrow(function: O, assumptions: IntCollection = EmptyCollection, guess: Instance? = null): Instance

    /**
     * Note that the iterator cannot be used in parallel, but multiple iterators can be used in parallel from the same
     * solver.
     */
    override fun iterator() = asSequence().iterator()

    /**
     * Note that the sequence cannot be used in parallel, but multiple sequences can be used in parallel from the same
     * solver. The method does not throw exceptions if
     * @param assumptions these variables will be fixed during solving, in dimacs format.
     */
    fun asSequence(assumptions: IntCollection = EmptyCollection): Sequence<Instance> {
        return generateSequence { witness(assumptions) }
    }

    /**
     * Set the random seed to a specific value to have a reproducible algorithm.
     */
    val randomSeed: Int
    /**
     * The solver will abort after timeout in milliseconds has been reached, without a real-time guarantee.
     */
    val timeout: Long

    val complete get() = false
}

interface OptimizerBuilder<in O : ObjectiveFunction> {

    /** Set the random seed to a specific value to have a reproducible algorithm. By default current system time. */
    fun randomSeed(randomSeed: Int): OptimizerBuilder<O>

    /** Set the timeout. */
    fun timeout(timeout: Long): OptimizerBuilder<O>

    fun build(): Optimizer<O>
}
