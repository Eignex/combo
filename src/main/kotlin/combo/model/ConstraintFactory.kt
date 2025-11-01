package combo.model

import combo.expressions.Cardinality
import combo.expressions.CardinalityVar
import combo.expressions.Conjunction
import combo.expressions.Constraint
import combo.expressions.Disjunction
import combo.expressions.Empty
import combo.expressions.IntVar
import combo.expressions.Linear
import combo.expressions.LinearVar
import combo.expressions.Proposition
import combo.expressions.PropositionalConstraint
import combo.expressions.ReifiedEquivalent
import combo.expressions.ReifiedImplies
import combo.expressions.Relation
import combo.math.gcd
import combo.math.gcdAll
import combo.expressions.Relation.*
import combo.expressions.Tautology
import combo.expressions.Value
import combo.util.*
import kotlin.math.absoluteValue

/**
 * The class is intended to be used as part of the [Model.Builder] and not be instantiated directly (although it can).
 * This class builds constraints in [CNF] form using with 0th order logic. It also contain some standard constraint
 * extensions where [CNF] form would be inefficient, for example cardinality and reification.
 */
@ModelMarker
class ConstraintFactory<S : Scope>(val scope: S, val index: VariableIndex) {

    infix fun Proposition.or(prop: Proposition) = or(this, prop)
    infix fun Proposition.and(prop: Proposition) = and(this, prop)
    infix fun Proposition.implies(prop: Proposition) = !this or prop
    infix fun Proposition.equivalent(prop: Proposition) = (this implies prop) and (prop implies this)
    infix fun Proposition.xor(prop: Proposition) = (this or prop) and (!this or !prop)

    infix fun Proposition.or(ref: String) = or(this, scope.resolve(ref))
    infix fun Proposition.and(ref: String) = and(this, scope.resolve(ref))
    infix fun Proposition.implies(ref: String) = !this or scope.resolve(ref)
    infix fun Proposition.equivalent(ref: String) = scope.resolve(ref).let { (this implies it) and (it implies this) }
    infix fun Proposition.xor(ref: String) = scope.resolve(ref).let { (this or it) and (!this or !it) }

    operator fun String.not() = scope.resolve(this).not()
    infix fun String.or(prop: Proposition) = or(scope.resolve(this), prop)
    infix fun String.and(prop: Proposition) = and(scope.resolve(this), prop)
    infix fun String.implies(prop: Proposition) = !scope.resolve(this) or prop
    infix fun String.equivalent(prop: Proposition) = with(scope.resolve(this)) { (this implies prop) and (prop implies this) }
    infix fun String.xor(prop: Proposition) = with(scope.resolve(this)) { (this or prop) and (!this or !prop) }

    infix fun String.or(ref: String) = or(scope.resolve(this), scope.resolve(ref))
    infix fun String.and(ref: String) = and(scope.resolve(this), scope.resolve(ref))
    infix fun String.implies(ref: String) = !scope.resolve(this) or scope.resolve(ref)
    infix fun String.equivalent(ref: String) = with(scope.resolve(this)) { scope.resolve(ref).let { (this implies it) and (it implies this) } }
    infix fun String.xor(ref: String) = with(scope.resolve(this)) { scope.resolve(ref).let { (this or it) and (!this or !it) } }

    /**
     * The constraint must be true if the value is true, but not vice versa.
     */
    infix fun Value.reifiedImplies(constraint: Constraint) =
        ReifiedImplies(toLiteral(index), constraint)

    /**
     * The constraint must be true if the value is true, and vice versa. The constraint must be a proposition which
     * can be negated.
     */
    infix fun Value.reifiedEquivalent(constraint: PropositionalConstraint) =
        ReifiedEquivalent(toLiteral(index), constraint)

    /**
     * The constraint must be true if the value is true, but not vice versa.
     */
    infix fun String.reifiedImplies(constraint: Constraint) =
        ReifiedImplies(scope.resolve(this).toLiteral(index), constraint)

    /**
     * The constraint must be true if the value is true, and vice versa. The constraint must be a proposition which
     * can be negated.
     */
    infix fun String.reifiedEquivalent(constraint: PropositionalConstraint) =
        ReifiedEquivalent(scope.resolve(this).toLiteral(index), constraint)

    /**
     * Any of the variable must be true, logical or.
     */
    fun disjunction(vararg variables: Value): PropositionalConstraint =
            if (variables.isEmpty()) Empty else Disjunction(toLiterals(variables))

    /**
     * All of the variable must be true, logical and.
     */
    fun conjunction(vararg variables: Value): PropositionalConstraint =
            if (variables.isEmpty()) Tautology else Conjunction(
                toLiterals(
                    variables
                )
            )

    /**
     * Specify a relation between the number of variables that are true.
     */
    fun cardinality(degree: Int, relation: Relation, vararg variables: Value): PropositionalConstraint {
        val literals = toLiterals(variables)
        if (relation.isTautology(0, literals.size, degree)) return Tautology
        if (relation.isEmpty(0, literals.size, degree)) return Empty
        return Cardinality(literals, degree, relation)
    }

    fun cardinality(degree: IntVar, relation: Relation, vararg variables: Value): PropositionalConstraint {
        val literals = toLiterals(variables)
        return CardinalityVar(
            literals,
            degree,
            index.valueIndexOf(degree),
            degree.parentLiteral(index),
            relation
        )
    }

    /**
     * Specify a relation with weights that are multiplied with variables.
     * x1*w1 + x2*w2 ... + xn*wn [relation] [degree],
     */
    fun linear(degree: Int, relation: Relation, weights: IntArray, variables: Array<out Value>): PropositionalConstraint {

        val gcd = gcd(degree.absoluteValue, gcdAll(*weights))
        val simplifiedDegree: Int
        val simplifiedWeights: IntArray
        if (gcd > 1) {
            simplifiedDegree = degree / gcd
            simplifiedWeights = weights.mapArray { it / gcd }
        } else {
            simplifiedDegree = degree
            simplifiedWeights = weights
        }

        val literals = IntHashMap()
        var k = 0
        variables.forEach { literals[it.toLiteral(index)] = k++ }

        val linear =
            Linear(literals, simplifiedWeights, simplifiedDegree, relation)
        if (relation.isTautology(linear.lowerBound, linear.upperBound, simplifiedDegree)) return Tautology
        if (relation.isEmpty(linear.lowerBound, linear.upperBound, simplifiedDegree)) return Empty
        return linear
    }

    fun linear(degree: IntVar, relation: Relation, weights: IntArray, variables: Array<out Value>): PropositionalConstraint {
        if (weights.isEmpty() || variables.isEmpty()) return Tautology

        val literals = IntHashMap()
        var k = 0
        variables.forEach { literals[it.toLiteral(index)] = k++ }

        return LinearVar(
            literals,
            weights,
            degree,
            index.valueIndexOf(degree),
            degree.parentLiteral(index),
            relation
        )
    }

    /**
     * Precisely this number of variables must be true.
     */
    fun exactly(degree: Int, vararg variables: Value) = cardinality(degree, EQ, *variables)

    /**
     * At most this number of variables must be true (defined with less than equal).
     */
    fun atMost(degree: Int, vararg variables: Value) = cardinality(degree, LE, *variables)

    /**
     * At least this number of variables must be true (defined with greater than equal).
     */
    fun atLeast(degree: Int, vararg variables: Value) = cardinality(degree, GE, *variables)

    /**
     * Declares variables to be mutually exclusive.
     */
    fun excludes(vararg variables: Value) = cardinality(1, LE, *variables)

    /**
     * Join propositions with logical or to [CNF] form.
     */
    fun or(vararg propositions: Proposition): Proposition {
        val literals = IntHashSet()
        var ands: ArrayList<CNF>? = null
        for (prop in propositions) {
            when (prop) {
                is Value -> literals.add(prop.toLiteral(index))
                is Disjunction -> literals.addAll(prop.literals)
                is Conjunction -> {
                    if (ands == null) ands = ArrayList()
                    ands.add(CNF(prop))
                }
                is CNF -> {
                    if (ands == null) ands = ArrayList()
                    ands.add(prop)
                }
                is Tautology -> return Tautology
                is Empty -> {
                }
                else -> throw UnsupportedOperationException("Cannot handle logic expression $prop.")
            }
        }
        val or: Proposition = if (literals.isEmpty()) Tautology else Disjunction(
            collectionOf(*literals.toArray())
        )
        return if (ands == null) {
            or
        } else {
            var result = CNF(ands[0].disjunctions)
            for (i in 1 until ands.size)
                result = result.distribute(ands[i])
            if (or is Disjunction)
                result = result.pullIn(or)
            if (result.disjunctions.size == 1) result.disjunctions[0]
            else result
        }
    }

    /**
     * Join propositions with logical and to [CNF] form.
     */
    fun and(vararg propositions: Proposition): Proposition {
        if (propositions.size <= 1) {
            return or(*propositions)
        }
        val disjunctions = ArrayList<Disjunction>()
        for (prop in propositions) {
            when (prop) {
                is Value -> disjunctions.add(
                    Disjunction(
                        collectionOf(
                            prop.toLiteral(
                                index
                            )
                        )
                    )
                )
                is Disjunction -> disjunctions.add(prop)
                is Conjunction -> prop.literals.forEach { disjunctions.add(
                    Disjunction(collectionOf(it))
                ) }
                is CNF -> disjunctions.addAll(prop.disjunctions)
                is Tautology -> return Tautology
                is Empty -> {
                }
                else -> throw UnsupportedOperationException("Cannot handle expression $prop.")
            }
        }
        for (d in disjunctions)
            if (d.size > 1) return CNF(disjunctions)

        // All disjunctions have size 1 so we convert to basic constraint conjunction
        val literals = IntArray(disjunctions.size)
        for ((i, d) in disjunctions.withIndex())
            literals[i] = d.literals.first()
        return Conjunction(collectionOf(*literals))
    }

    private fun toLiterals(vars: Array<out Value>): IntCollection {
        // Remove any duplicates
        val set = IntHashSet()
        vars.forEach { set.add(it.toLiteral(index)) }
        return collectionOf(*set.toArray())
    }
}
