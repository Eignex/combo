package combo.expressions

import combo.sat.*
import combo.util.*
import kotlin.math.min
import kotlin.random.Random

sealed class ReifiedConstraint(
    val literal: Int,
    open val constraint: Constraint
) :
    Constraint {

    override val priority: Int get() = constraint.priority + literals.size

    override fun cacheUpdate(cacheResult: Int, newLit: Int) =
        if (newLit.toIx() == literal.toIx()) cacheResult
        else constraint.cacheUpdate(cacheResult, newLit)

    override fun cache(instance: Instance) = constraint.cache(instance)
    override fun isUnit() = false
}

/**
 * ReifiedEquivalent encodes the constraint [literal] <=> [constraint].
 * That is, the constraint is satisfied when both the [constraint] and
 * [literal] is satisfied or when neither of them are.
 */
class ReifiedEquivalent(
    literal: Int,
    override val constraint: PropositionalConstraint
) : ReifiedConstraint(literal, constraint),
    PropositionalConstraint {

    init {
        assert(constraint.literals.isNotEmpty())
        assert(literal !in constraint.literals && !literal !in constraint.literals)
    }

    override val literals: IntCollection =
        unionCollection(constraint.literals, literal)

    override operator fun not() = ReifiedEquivalent(!literal, constraint)

    override fun violations(instance: Instance, cacheResult: Int): Int {
        val constraintViolations = constraint.violations(instance, cacheResult)
        return if (instance.literal(literal.toIx()) == literal) min(
            1,
            constraintViolations
        )
        else return if (constraintViolations == 0) 1 else 0
    }

    override fun unitPropagation(unit: Int): PropositionalConstraint {
        return when (unit) {
            literal -> constraint
            !literal -> constraint.not()
            else -> {
                when (val propagated = constraint.unitPropagation(unit)) {
                    constraint -> this
                    is Tautology -> Conjunction(collectionOf(literal))
                    is Empty -> Conjunction(collectionOf(!literal))
                    else -> ReifiedEquivalent(literal, propagated)
                }
            }
        }
    }

    override fun coerce(instance: Instance, rng: Random) {
        if (instance.literal(literal.toIx()) == literal) constraint.coerce(
            instance,
            rng
        )
        else if (constraint.satisfies(instance)) instance.set(literal)
    }

    fun toCnf(): Sequence<Disjunction> {
        return when (constraint) {
            is Disjunction -> {
                val c1 = constraint.literals.asSequence()
                    .map { Disjunction(collectionOf(literal, !it)) }
                val c2 = sequenceOf(
                    Disjunction(
                        constraint.literals.mutableCopy(nullValue = 0)
                            .apply { add(!literal) })
                )
                c1 + c2
            }

            is Conjunction -> {
                val c1 = constraint.literals.asSequence()
                    .map { Disjunction(collectionOf(!literal, it)) }
                val c2 = sequenceOf(
                    Disjunction(
                        (constraint.literals.mutableCopy(nullValue = 0)
                            .map { !it }.apply { add(literal) })
                    )
                )
                c1 + c2
            }

            else -> throw IllegalArgumentException("Cannot convert arbitrary constraint to CNF.")
        }
    }

    override fun toString() = "ReifiedEquivalent($literal <=> $constraint)"
}

/**
 * ReifiedImplies encodes the constraint [literal] => [constraint].
 * This is also known as the IfThen constraint, i.e. * if [literal] then [constraint].
 */
class ReifiedImplies(literal: Int, constraint: Constraint) :
    ReifiedConstraint(literal, constraint) {

    init {
        assert(constraint.literals.isNotEmpty())
        assert(literal !in constraint.literals && !literal !in constraint.literals)
    }

    override val literals: IntCollection =
        unionCollection(constraint.literals, literal)

    override fun violations(instance: Instance, cacheResult: Int): Int {
        return if (instance.literal(literal.toIx()) == literal) min(
            1,
            constraint.violations(instance, cacheResult)
        )
        else 0
    }

    override fun unitPropagation(unit: Int): Constraint {
        return when {
            unit == literal -> constraint
            unit.toIx() == literal.toIx() -> Tautology
            else -> {
                when (val propagated = constraint.unitPropagation(unit)) {
                    constraint -> this
                    is Tautology -> Tautology
                    is Empty -> Conjunction(collectionOf(!literal))
                    else -> ReifiedImplies(literal, propagated)
                }
            }
        }
    }

    override fun coerce(instance: Instance, rng: Random) {
        if (instance.literal(literal.toIx()) == literal) constraint.coerce(
            instance,
            rng
        )
    }

    fun toCnf(): Sequence<Disjunction> {
        return when (constraint) {
            is Disjunction -> sequenceOf(
                Disjunction(
                    IntUnionCollection(
                        collectionOf(!literal),
                        constraint.literals
                    )
                )
            )

            is Conjunction -> constraint.literals.asSequence()
                .map { Disjunction(collectionOf(!literal, it)) }

            else -> throw IllegalArgumentException("Cannot convert arbitrary constraint to CNF.")
        }
    }

    override fun toString() = "ReifiedImplies($literal => $constraint)"
}
