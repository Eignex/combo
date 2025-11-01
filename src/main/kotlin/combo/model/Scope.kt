package combo.model

import combo.expressions.Value
import combo.expressions.Variable
import combo.util.ArrayQueue


class ChildScope<out D : Scope> internal constructor(override val scopeName: String, override val reifiedValue: Value, override val parent: D) : Scope {
    override fun toString() = "ChildScope($scopeName)"
    override val children: List<ChildScope<ChildScope<D>>> = ArrayList()
    override val names: Map<String, Variable<*, *>> = HashMap()
    override val variables: List<Variable<*, *>> = ArrayList()
    override val isRoot: Boolean get() = false
    override fun addScope(scopeName: String, reifiedValue: Value) = ChildScope(scopeName, reifiedValue, this).also {
        (children as MutableList).add(it)
    }
}

class RootScope(override val reifiedValue: Value) : Scope {
    override fun toString() = "RootScope($scopeName)"
    override val scopeName get() = reifiedValue.name
    override val parent: Nothing get() = throw NoSuchElementException()
    override val children: List<ChildScope<RootScope>> = ArrayList()
    override val names: Map<String, Variable<*, *>> = HashMap()
    override val variables: List<Variable<*, *>> = ArrayList()
    override val isRoot: Boolean get() = true
    override fun addScope(scopeName: String, reifiedValue: Value) = ChildScope(scopeName, reifiedValue, this).also {
        (children as MutableList).add(it)
    }
}

/**
 * This can be used to query the variables and child scopes defined as part of the model.
 * The [find] and [get] method searches for variables in sub-scopes, [resolve] searches up in the hierarchy.
 */
interface Scope : Iterable<Variable<*, *>> {

    val scopeName: String
    val reifiedValue: Value
    val parent: Scope
    val children: List<Scope>
    val names: Map<String, Variable<*, *>>
    val variables: List<Variable<*, *>>
    val isRoot: Boolean

    /**
     * Performs a breadth first search from the current scope and down, throwing an error in case of failure.
     */
    operator fun get(name: String): Variable<*, *> = find(name)
            ?: throw NoSuchElementException("Scope contains no variable with name $name.")

    /**
     * Performs a breadth first search from the current scope and down, throwing an error in case of failure.
     * Then the [value] is extracted from the value.
     */
    operator fun <V> get(name: String, value: V) = find<Variable<V, *>>(name)?.value(value)
            ?: throw NoSuchElementException("Scope contains no variable with name $name.")

    /**
     * See [resolve].
     */
    fun inScope(reference: String): Boolean {
        var p: Scope = this
        while (true) {
            if (reference in p.names) return true
            if (p.isRoot) return false
            p = p.parent
        }
    }

    /**
     * Resolve the [reference] in the current context, moving upwards in the symbol tree hierarchy if a match is not
     * found. This function is intended for internal use through [ConstraintFactory], which is why it throws an
     * exception on failure.
     */
    fun resolve(reference: String): Variable<*, *> {
        var p: Scope = this
        while (true) {
            val r = p.names[reference]
            if (r != null) return r
            if (p.isRoot) throw NoSuchElementException("Scope contains no variable in scope with name $reference.")
            p = p.parent
        }
    }

    /**
     * Performs a breadth-first search for resolving a variable. This is suitable to use at the root of the tree
     * hierarchy to find the closest child variable.
     */
    @Suppress("UNCHECKED_CAST")
    fun <V : Variable<*, *>> find(reference: String): V? {
        run {
            val v = names[reference]
            if (v != null) return v as V
        }

        val queue = ArrayQueue<Scope>()
        queue.add(this)

        while (queue.size > 0) {
            val vi = queue.remove()
            val v = vi.names[reference]
            if (v != null) return v as V
            else queue.addAll(vi.children)
        }

        return null
    }

    operator fun contains(reference: String): Boolean = find<Variable<*, *>>(reference) != null

    /**
     * All variables in the current scope and all scopes below this, including all child scopes. The sequence is
     * iterated in depth-first order using a stack.
     */
    fun asSequence(): Sequence<Variable<*, *>> = asSequenceWithScope().map { it.first }

    fun asSequenceWithScope(): Sequence<Pair<Variable<*, *>, Scope>> {
        val stack = ArrayList<Scope>()

        var i = 0
        var current: Scope = this

        run {
            val currentVars = current.variables
            while (i < currentVars.size && currentVars[i].nbrValues <= 0)
                i++
            stack.addAll(this.children)
        }

        return generateSequence {
            var scopeVars = current.variables
            while (i < scopeVars.size && scopeVars[i].nbrValues <= 0)
                i++

            while (i >= current.variables.size && stack.isNotEmpty()) {
                current = stack.removeAt(stack.lastIndex)
                stack.addAll(current.children)
                i = 0
                scopeVars = current.variables
                // Advance over empty variables (unit and reference)
                while (i < scopeVars.size && scopeVars[i].nbrValues <= 0)
                    i++
            }

            // Return next variable or terminate with null
            scopeVars = current.variables
            if (i < scopeVars.size) scopeVars[i++] to current
            else null
        }
    }

    fun scopesAsSequence(): Sequence<Scope> {
        val stack = ArrayList<Scope>()
        stack.add(this)
        return generateSequence {
            if (stack.isEmpty()) null
            else {
                stack.removeAt(stack.lastIndex).also {
                    stack.addAll(it.children)
                }
            }
        }
    }

    fun add(variable: Variable<*, *>) {
        assert(variable.nbrValues > 0)
        require(!names.containsKey(variable.name)) {
            "Variable with name ${variable.name} already exists in scope $scopeName."
        }
        (names as MutableMap)[variable.name] = variable
        (variables as MutableList).add(variable)
    }

    fun addScope(scopeName: String, reifiedValue: Value): Scope

    override fun iterator() = asSequence().iterator()

}
