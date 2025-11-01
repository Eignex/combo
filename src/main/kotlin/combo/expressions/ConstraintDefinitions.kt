package combo.expressions

import combo.sat.*
import combo.util.*
import kotlin.math.min
import kotlin.random.Random


/**
 * A constraint must be satisfied during solving.
 * See [Literal] for more information on the binary format of variables.
 */
sealed interface Constraint : Expression {

    val literals: IntCollection
    val size get() = literals.size

    val priority: Int

    fun isUnit(): Boolean = size == 1
    fun unitLiterals(): IntArray = literals.toArray()

    /**
     * Returns the number of changes necessary for the constraint to be
     * satisfied, based on a cached result.
     */
    fun violations(instance: Instance, cacheResult: Int): Int

    /**
     * Update the cached result with the changing literal [newLit].
     * This method can only be called if the literal is contained in [literals].
     */
    fun cacheUpdate(cacheResult: Int, newLit: Int) =
        cacheResult + if (newLit in literals) 1 else -1

    /**
     * Calculate the cached result of satisfy value. This will be updated with
     * the [cacheUpdate] and used in the [violations] method.
     * The default implementation gathers the number of matching literals.
     */
    fun cache(instance: Instance): Int {
        var sum = 0
        if (literals is IntRangeCollection) {
            val lits = literals as IntRangeCollection
            var ix = min(lits.min.toIx(), lits.max.toIx())
            val ints = (size shr 5) + if (size and 0x1F > 0) 1 else 0
            for (i in 0 until ints) {
                val nbrBits =
                    if (i == ints - 1) ((size - 1) and 0x1F) + 1 else 32
                val value = instance.getBits(ix, nbrBits)
                sum += if (lits.min < 0) nbrBits - Int.bitCount(value) else Int.bitCount(
                    value
                )
                ix += nbrBits
            }
            return sum
        }
        for (lit in literals.iterator()) {
            if (instance.literal(lit.toIx()) == lit) sum++
        }
        return sum
    }

    /**
     * Returns the number of changes necessary for the constraint to be satisfied.
     * This method is only used for test and debug,
     * in actual solving the cached version is used instead.
     */
    fun violations(instance: Instance): Int =
        violations(instance, cache(instance))

    /**
     * Returns whether the constraint satisfies the instance.
     */
    fun satisfies(instance: Instance) = violations(instance) == 0

    /**
     * Change the constraint based on the value of a unit literal.
     */
    fun unitPropagation(unit: Int): Constraint

    fun coerce(instance: Instance, rng: Random)
}

/**
 * A logic constraint can be negated "for free" without increasing the cost
 * solving.
 */
interface PropositionalConstraint : Proposition, Constraint {
    override fun unitPropagation(unit: Int): PropositionalConstraint = this
    override fun not(): PropositionalConstraint
}

/**
 * This constraint is never satisfied.
 */
object Empty : PropositionalConstraint, Proposition {
    override val priority: Int get() = 0
    override val literals get() = EmptyCollection
    override fun violations(instance: Instance, cacheResult: Int) =
        Int.MAX_VALUE

    override operator fun not() = Tautology
    override fun unitPropagation(unit: Int) = this
    override fun coerce(instance: Instance, rng: Random) {}
    override fun toString() = "Empty"
}

/**
 * This constraint is always satisfied.
 */
object Tautology : PropositionalConstraint, Proposition {
    override val priority: Int get() = 0
    override val literals get() = EmptyCollection
    override fun violations(instance: Instance, cacheResult: Int) = 0
    override operator fun not() = Empty
    override fun unitPropagation(unit: Int) = this
    override fun coerce(instance: Instance, rng: Random) {}
    override fun toString() = "Tautology"
}
