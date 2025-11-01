package combo.sat

import combo.expressions.Constraint
import combo.expressions.Empty
import combo.expressions.Tautology
import combo.expressions.Conjunction
import combo.util.*
import kotlin.jvm.JvmOverloads

/**
 * This class sufficiently describes a constraint satisfaction problem.
 * @param nbrValues Number of binary variables the problem has.
 * @param constraints All constraints that the [Instance]s will be satisfied on.
 */
class Problem @JvmOverloads constructor(val nbrValues: Int, val constraints: Array<out Constraint> = emptyArray()) {

    val nbrConstraints get() = constraints.size

    private interface ConstraintIndex {
        fun constraining(valueIx: Int): IntArray
    }

    private class MapConstraintIndex(val index: Map<Int, IntArray>) : ConstraintIndex {
        constructor(nbrValues: Int, constraints: Array<out Constraint>) : this(
                constraints.let {
                    val map = HashMap<Int, IntArrayList>()
                    for ((i, cons) in it.withIndex()) {
                        for (lit in cons.literals) {
                            val ix = lit.toIx()
                            assert(ix < nbrValues)
                            if (!map.containsKey(ix)) map[ix] = IntArrayList()
                            map[ix]!!.add(i)
                        }
                    }
                    map.mapValuesTo(HashMap()) { it.value.toArray() }
                }
        )

        override fun constraining(valueIx: Int) = index[valueIx] ?: EMPTY_INT_ARRAY
    }

    private class ArrayConstraintIndex(val index: Array<IntArray>) : ConstraintIndex {
        constructor(nbrValues: Int, constraints: Array<out Constraint>) : this(
                constraints.let { cs ->
                    val index: Array<IntArrayList?> = arrayOfNulls(nbrValues)
                    for ((i, cons) in cs.withIndex()) {
                        for (lit in cons.literals) {
                            val ix = lit.toIx()
                            assert(ix < nbrValues)
                            if (index[ix] == null) index[ix] = IntArrayList()
                            index[ix]!!.add(i)
                        }
                    }
                    Array<IntArray>(nbrValues) {
                        if (index[it] == null) EMPTY_INT_ARRAY
                        else index[it]!!.toArray()
                    }
                }
        )

        override fun constraining(valueIx: Int) = index[valueIx]
    }

    private val constraintIndex: ConstraintIndex = if (nbrValues < 10000 && constraints.size * 10 > nbrValues)
        ArrayConstraintIndex(nbrValues, constraints)
    else MapConstraintIndex(nbrValues, constraints)

    /**
     * Returns the index into the [constraints] array of all constraints with the given variable.
     */
    fun constraining(valueIx: Int) = constraintIndex.constraining(valueIx)

    /**
     * This method is intended for testing, solvers use [Validator] instead.
     * @return true if all constraints are satisfied.
     */
    fun satisfies(instance: Instance) = constraints.all { it.satisfies(instance) }

    /**
     * This method is intended for testing, solvers use [Validator] instead.
     * @return sum of all constraint violations, this will be 0 if the problem is satisfied.
     */
    fun violations(instance: Instance) =
        constraints.sumOf { it.violations(instance) }

    /**
     * Performs unit propagation on all constraints. Additional unit variables can be added in the [unitLiterals] parameter.
     * New unitLiterals will be added to the set. This method does not change the original problem but can return
     * propagated constraints.
     */
    @JvmOverloads
    fun unitPropagation(unitLiterals: IntHashSet = IntHashSet(), returnConstraints: Boolean = false): Array<Constraint> {

        fun addUnit(units: IntHashSet, unit: Int): Boolean {
            if (units.contains(!unit)) throw UnsatisfiableException("Unsatisfiable by unit propagation.", literal = unit)
            else return units.add(unit)
        }

        val unitConstraint = ArrayList<Int>()
        if (unitLiterals.isNotEmpty())
            unitConstraint.add(constraints.size + 1)

        val initial: Constraint = if (unitLiterals.isEmpty()) Empty else Conjunction(unitLiterals.copy())
        for (i in constraints.indices)
            if (constraints[i].isUnit()) {
                unitConstraint.add(i)
                for (l in constraints[i].unitLiterals())
                    addUnit(unitLiterals, l)
            }

        val copy = Array(constraints.size) { constraints[it] }

        while (!unitConstraint.isEmpty()) {
            val constraintId = unitConstraint.removeAt(0)
            val constraint = if (constraintId >= constraints.size) initial else copy[constraintId]
            for (unitLit in constraint.unitLiterals()) {
                val unitId = unitLit.toIx()
                val matching = constraining(unitId)
                for (i in matching.indices) {
                    if (i == constraintId) continue
                    val constraintI = copy[matching[i]]
                    val reduced = constraintI.unitPropagation(unitLit)
                    if (reduced is Empty)
                        throw UnsatisfiableException("Unsatisfiable by unit propagation.")
                    copy[matching[i]] = reduced
                    if (reduced.isUnit()) {
                        var anyNew = false
                        for (lit in reduced.unitLiterals())
                            anyNew = addUnit(unitLiterals, lit) || anyNew
                        if (anyNew)
                            unitConstraint.add(matching[i])
                    }
                }
            }
        }
        return if (returnConstraints) {
            // TODO takes actual time
            copy.asSequence()
                    .filter { !it.isUnit() && it != Tautology }
                    .toList()
                    .toTypedArray()
        } else emptyArray()
    }
}
