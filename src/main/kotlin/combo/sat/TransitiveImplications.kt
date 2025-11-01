package combo.sat

import combo.expressions.Cardinality
import combo.expressions.Conjunction
import combo.expressions.Disjunction
import combo.expressions.ReifiedEquivalent
import combo.expressions.Relation
import combo.util.IntArrayList
import combo.util.IntHashSet
import combo.util.isNotEmpty
import kotlin.math.min

/**
 * This class calculates the transitive closure of a graph in adjacency list form.
 * This is a recursion-free implementation of the Simple_SC algorithm by Esko Nuutilas,
 * see http://www.cs.hut.fi/~enu/thesis.html
 */
class TransitiveImplications(val nbrValues: Int, implications: Map<Int, IntArray>) {

    /**
     * This constructor calculates the initial edges in the graph.
     */
    constructor(problem: Problem) : this(problem.nbrValues, HashMap<Int, IntHashSet>().let { map ->
        for (constraint in problem.constraints) {
            if (constraint is Disjunction && constraint.size == 2) {
                val array = constraint.literals.toArray()
                map.getOrPut(!array[0]) { IntHashSet() }.add(array[1])
                map.getOrPut(!array[1]) { IntHashSet() }.add(array[0])
            } else if ((constraint is Cardinality && constraint.degree == 1 && constraint.relation == Relation.LE) ||
                    (constraint is Cardinality && constraint.degree == 2 && constraint.relation == Relation.LT)) {
                val array = constraint.literals.toArray()
                for (i in array.indices) {
                    for (j in (i + 1) until array.size) {
                        map.getOrPut(array[i]) { IntHashSet() }.add(!array[j])
                        map.getOrPut(array[j]) { IntHashSet() }.add(!array[i])
                    }
                }
            } else if (constraint is ReifiedEquivalent && constraint.constraint is Disjunction) {
                for (literal in constraint.constraint.literals) {
                    map.getOrPut(literal) { IntHashSet() }.add(constraint.literal)
                    map.getOrPut(!constraint.literal) { IntHashSet() }.add(!literal)
                }
            } else if (constraint is ReifiedEquivalent && constraint.constraint is Conjunction) {
                for (literal in constraint.constraint.literals) {
                    map.getOrPut(!literal) { IntHashSet() }.add(!constraint.literal)
                    map.getOrPut(constraint.literal) { IntHashSet() }.add(literal)
                }
            }
        }
        map.mapValues { it.value.toArray() }
    })

    // Using a different encoding here for literals, so that literal can be used as array index.
    // -1 = 0
    // 1  = 1
    // -2 = 2
    // 2  = 3
    private val trueImplications = arrayOfNulls<SparseBitArray?>(nbrValues * 2)
    private val falseImplications = arrayOfNulls<SparseBitArray?>(nbrValues * 2)

    fun trueImplications(literal: Int) = trueImplications[toArrayIndex(literal)]
    fun falseImplications(literal: Int) = falseImplications[toArrayIndex(literal)]

    fun propagate(literal: Int, instance: Instance) {
        trueImplications(literal)?.run { instance.or(this) }
        falseImplications(literal)?.run { instance.andNot(this) }
    }

    fun hasPropagations(literal: Int): Boolean {
        val ix = toArrayIndex(literal)
        return trueImplications[ix] != null || falseImplications[ix] != null
    }

    fun toArray(literal: Int): IntArray {
        val set = IntHashSet()
        trueImplications[toArrayIndex(literal)]?.forEach { set.add(it.toLiteral(true)) }
        falseImplications[toArrayIndex(literal)]?.forEach { set.add(it.toLiteral(false)) }
        return set.toArray()
    }

    fun flipPropagate(instance: Instance, ix: Int) {
        instance.flip(ix)
        val literal = instance.literal(ix)
        val trueImplications = trueImplications(literal)
        if (trueImplications != null)
            for (j in trueImplications) instance[j] = true
        val falseImplications = falseImplications(literal)
        if (falseImplications != null)
            for (j in falseImplications) instance[j] = false
    }

    init {
        for ((from, edges) in implications) {
            val ix = toArrayIndex(from)
            for (to in edges) {
                if (to.toBoolean()) {
                    if (trueImplications[ix] == null) trueImplications[ix] = SparseBitArray(nbrValues)
                    trueImplications[ix]!![to.toIx()] = true
                } else {
                    if (falseImplications[ix] == null) falseImplications[ix] = SparseBitArray(nbrValues)
                    falseImplications[ix]!![to.toIx()] = true
                }
            }
        }

        val visited = BooleanArray(nbrValues * 2)
        val component = IntArray(nbrValues * 2) { -1 }
        val root = IntArray(nbrValues * 2)

        var components = 0
        val stack = IntArrayList()

        // This is a stack simulation to avoid recursion and stack overflow for large graphs (n>10000)
        val callStack = IntArrayList()

        for (v in 0 until nbrValues * 2) {
            if (visited[v]) continue
            callStack.add(v)
            callStack.add(0)
            while (callStack.isNotEmpty()) {
                val i = callStack.removeAt(callStack.size - 1)
                val v1 = callStack.removeAt(callStack.size - 1)
                if (i == 0) {
                    visited[v1] = true
                    root[v1] = v1
                    stack.add(v1)
                }
                var recurse = false
                for (j in i until (implications[toLiteral(v1)]?.size ?: 0)) {
                    val w = toArrayIndex(implications[toLiteral(v1)]!![j])
                    if (!visited[w]) {
                        callStack.add(v1)
                        callStack.add(j + 1)
                        callStack.add(w)
                        callStack.add(0)
                        recurse = true
                        break
                    }
                    if (component[w] < 0) root[v1] = min(root[v1], root[w])
                    addImplications(v1, w)
                }
                if (recurse) continue
                if (root[v1] == v1) {
                    var w: Int
                    do {
                        w = stack.removeAt(stack.size - 1)
                        component[w] = components

                        val ti = trueImplications[v1]
                        val fi = falseImplications[v1]
                        trueImplications[w] = ti
                        falseImplications[w] = fi

                        // If a variable and its negation belong to the same strongly connected component, ie. x <=> !x
                        if (component[w] == component[w xor 1])
                            throw UnsatisfiableException("Unsatisfiable during initialization of implication graph (2-sat).")

                    } while (w != v1)
                    components++
                }
                if (callStack.isNotEmpty()) {
                    val w = v1
                    val v2 = callStack[callStack.size - 2]
                    root[v2] = min(root[v2], root[w])
                    addImplications(v2, w)
                }
            }
        }
    }

    private fun addImplications(to: Int, from: Int) {
        if (trueImplications[to] == null && trueImplications[from] != null) trueImplications[to] = SparseBitArray(nbrValues)
        if (trueImplications[from] != null) trueImplications[to]!!.or(trueImplications[from]!!)
        if (falseImplications[to] == null && falseImplications[from] != null) falseImplications[to] = SparseBitArray(nbrValues)
        if (falseImplications[from] != null) falseImplications[to]!!.or(falseImplications[from]!!)
    }

    private fun toArrayIndex(literal: Int) = (literal.toIx() shl 1) + if (literal.toBoolean()) 0 else 1
    private fun toLiteral(arrayIndex: Int) = (arrayIndex shr 1).toLiteral(arrayIndex and 1 == 0)
}
