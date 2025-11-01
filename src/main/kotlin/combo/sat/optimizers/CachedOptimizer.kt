@file:JvmName("CachedSolvers")

package combo.sat.optimizers

import combo.expressions.Constraint
import combo.expressions.Tautology
import combo.sat.*
import combo.expressions.Conjunction
import combo.util.IntCollection
import combo.util.RandomListCache
import combo.util.RandomSequence
import combo.util.isEmpty
import kotlin.jvm.JvmName
import kotlin.jvm.JvmOverloads

/**
 * A cached optimizer reuses previous solutions, provided that there are previous instances that match the assumptions.
 * As with all solvers, this can be used in parallel by multiple threads.
 * Using a cache with multiple threads in parallel makes the solver non-deterministic even if a specific random seed
 * is used. Sequence and iteration will not be cached.
 * @param baseOptimizer Optimizer that is called on cache miss.
 * @param maxSize Maximum number of instances kept in the cache.
 * @param pNew Chance of generating new instance regardless of whether there are any instances matching the assumptions.
 * @param pNewWithGuess Chance of generating new instance using a guess initial solution randomly selected from a matching instance.
 */
class CachedOptimizer<in O : ObjectiveFunction> @JvmOverloads constructor(
        val baseOptimizer: Optimizer<O>,
        val maxSize: Int = 20,
        val pNew: Float = 0.0f,
        val pNewWithGuess: Float = 0.1f
) : Optimizer<O> {

    override val randomSeed get() = baseOptimizer.randomSeed
    override val timeout get() = baseOptimizer.timeout
    private val randomSequence = RandomSequence(randomSeed)

    private val buffer = RandomListCache<Instance>(maxSize, randomSeed)

    override fun optimizeOrThrow(function: O, assumptions: IntCollection, guess: Instance?): Instance {
        val c: Constraint = if (assumptions.isEmpty()) Tautology else Conjunction(assumptions)
        val rng = randomSequence.next()
        var minV = Float.MAX_VALUE
        var best: Instance? = null
        var failure: ValidationException? = null
        var new = false

        if (rng.nextFloat() < pNew) {
            new = true
            try {
                best = baseOptimizer.optimizeOrThrow(function, assumptions, guess)
                buffer.put(best)
                minV = function.value(best)
            } catch (e: ValidationException) {
                failure = e
            }
        }

        buffer.forEach {
            if (c.satisfies(it)) {
                val v = function.value(it)
                if (v < minV) {
                    minV = v
                    best = it
                }
            }
        }

        if (!new && minV > function.lowerBound() && best != null && rng.nextFloat() < pNewWithGuess) {
            try {
                val guessed = baseOptimizer.optimizeOrThrow(function, assumptions, best!!.copy())
                val v = function.value(guessed)
                if (v != minV)
                    buffer.put(guessed)
                if (v < minV)
                    best = guessed
            } catch (e: ValidationException) {
                failure = e
            }
        }
        return if (best == null && failure == null)
            baseOptimizer.optimizeOrThrow(function, assumptions, guess).also { buffer.put(it) }
        else best ?: throw UnsatisfiableException("Failed to find matching fallback instance.", failure)
    }

    override fun witnessOrThrow(assumptions: IntCollection, guess: Instance?): Instance {
        val rng = randomSequence.next()
        var failure: ValidationException? = null
        try {
            if (rng.nextFloat() < pNew)
                return baseOptimizer.witnessOrThrow(assumptions, guess).also {
                    buffer.put(it)
                }
        } catch (e: ValidationException) {
            failure = e
        }
        val c: Constraint = if (assumptions.isEmpty()) Tautology else Conjunction(assumptions)
        val l: Instance? = buffer.find { c.satisfies(it) }
        return if (l == null && failure == null)
            baseOptimizer.witnessOrThrow(assumptions, guess).also { buffer.put(it) }
        else l ?: throw UnsatisfiableException("Failed to find matching fallback instance.", failure)
    }

    override fun asSequence(assumptions: IntCollection) = baseOptimizer.asSequence(assumptions)

    class Builder<O : ObjectiveFunction>(val optimizer: Optimizer<O>) : OptimizerBuilder<O> {
        private var maxSize: Int = 20
        private var pNew: Float = 0.05f
        private var pNewWithGuess: Float = 1.0f

        /** Maximum number of instances kept in the cache. */
        fun maxSize(maxSize: Int) = apply { this.maxSize = maxSize }

        /** Chance of generating new instance regardless of whether there are any instances matching the assumptions. */
        fun pNew(pNew: Float) = apply { this.pNew = pNew }

        /** Chance of generating new instance using a guess initial solution randomly selected from a matching instance. */
        fun pNewWithGuess(pNewWithGuess: Float) = apply { this.pNewWithGuess = pNewWithGuess }

        override fun build() = CachedOptimizer(optimizer, maxSize, pNew, pNewWithGuess)
        override fun randomSeed(randomSeed: Int) = error("Set on wrapping optimizer before building. TODO this contain builder instead.")
        override fun timeout(timeout: Long) = error("Set on wrapping optimizer before building.")
    }
}
