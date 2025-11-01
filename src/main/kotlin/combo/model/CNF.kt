package combo.model

import combo.sat.*
import combo.expressions.Conjunction
import combo.expressions.Disjunction
import combo.expressions.Empty
import combo.expressions.Proposition
import combo.expressions.PropositionalConstraint
import combo.expressions.Tautology
import combo.util.IntHashSet
import combo.util.collectionOf
import combo.util.isEmpty

class CNF(val disjunctions: List<Disjunction>) : Proposition {

    constructor(conjunction: Conjunction) : this(ArrayList<Disjunction>().also {
        conjunction.literals.forEach { lit -> it.add(Disjunction(collectionOf(lit))) }
    })

    fun pullIn(to: Disjunction): CNF {
        val result = ArrayList<Disjunction>()
        for (o in disjunctions) {
            val set = IntHashSet()
            set.addAll(o.literals)
            set.addAll(to.literals)
            val dis = validateDisjunction(set)
            when (dis) {
                is Empty -> throw UnsatisfiableException()
                is Tautology -> {
                }
                is Disjunction -> result.add(dis)
            }
        }
        return CNF(result)
    }

    fun distribute(cnf: CNF): CNF {
        val result = ArrayList<Disjunction>()
        for (o1 in disjunctions)
            for (o2 in cnf.disjunctions) {
                val set = IntHashSet()
                set.addAll(o1.literals)
                set.addAll(o2.literals)
                val dis = validateDisjunction(set)
                when (dis) {
                    is Empty -> throw UnsatisfiableException()
                    is Tautology -> {
                    }
                    is Disjunction -> result.add(dis)
                }
            }
        return CNF(result)
    }

    private fun validateDisjunction(set: IntHashSet): PropositionalConstraint {
        if (set.isEmpty()) return Empty
        for (lit in set) {
            if (!lit in set) return Tautology
        }
        return Disjunction(collectionOf(*set.toArray()))
    }

    override operator fun not() = disjunctions.asSequence()
            .map { CNF(!it) }
            .reduce { a: CNF, b: CNF -> a.distribute(b) }
}
