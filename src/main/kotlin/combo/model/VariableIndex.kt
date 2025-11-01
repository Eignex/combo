package combo.model

import combo.expressions.Root
import combo.expressions.Variable


/**
 * Contains mappings from variable to the index they have in optimization problems.
 * Variables can be iterated in declaration order.
 */
class VariableIndex : Iterable<Variable<*, *>> {

    private val index: MutableMap<Variable<*, *>, Int> = LinkedHashMap()
    private val variables = ArrayList<Variable<*, *>>()

    var nbrValues: Int = 0
        private set

    val nbrVariables: Int get() = index.size

    fun add(variable: Variable<*, *>): Int {
        assert(variable !is Root)
        require(!index.containsKey(variable)) { "Variable $variable already added." }
        index[variable] = nbrValues
        variables.add(variable)
        nbrValues += variable.nbrValues
        return nbrValues
    }

    fun variable(variableIndex: Int) = variables[variableIndex]
    fun indexOf(variable: Variable<*, *>) = variables.indexOf(variable)

    fun variableWithValue(valueIndex: Int): Variable<*, *> {
        var l = 0
        var r = nbrVariables
        while (l <= r) {
            val m = l + (r - l) / 2
            val ix = valueIndexOf(variables[m])
            if (valueIndex in ix..ix + variables[m].nbrValues) return variables[m]
            if (ix < valueIndex) l = m + 1
            else r = m - 1
        }
        throw NoSuchElementException("Variable with valueIndex $valueIndex not found.")
    }

    fun valueIndexOf(variable: Variable<*, *>): Int =
            if (variable is Root) throw IllegalArgumentException("Root variable cannot be referenced.")
            else index[variable] ?: throw NoSuchElementException("Variable $variable not found.")

    operator fun contains(variable: Variable<*, *>): Boolean = index.containsKey(variable)

    override fun iterator(): Iterator<Variable<*, *>> = index.keys.iterator()
}
