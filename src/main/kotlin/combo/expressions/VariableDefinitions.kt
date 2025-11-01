package combo.expressions

import combo.model.Scope
import combo.model.VariableIndex
import combo.sat.Instance
import combo.sat.toLiteral
import java.util.concurrent.atomic.AtomicInteger

/**
 * This class represents the decision variable in the combinatorial optimization
 * problem. They must be registered in the [combo.model.Model] to be used.
 * The easiest way of constructing them is through the various methods in
 * [combo.model.Model.Builder], such as [combo.model.Model.Builder.flag] or
 * [combo.model.Model.Builder.nominal] which will also add the required
 * constraints.
 * @param V the type that sub options are parameterized by.
 * @param T the type that is returned, often same as [V].
 */
abstract class Variable<in V, out T>(override val name: String) : Value {

    companion object {
        fun defaultName() = $$"$x_$${COUNTER.getAndIncrement()}"
        private val COUNTER: AtomicInteger = AtomicInteger()
    }

    override fun toLiteral(variableIndex: VariableIndex) =
        if (optional) variableIndex.valueIndexOf(this).toLiteral(true)
        else parent.toLiteral(variableIndex)

    fun parentLiteral(variableIndex: VariableIndex) =
        if (parent is Root) 0
        else parent.toLiteral(variableIndex)

    /**
     * The reified value is the value that governs whether the variable is set,
     * it is usually itself or [Root].
     */
    override val canonicalVariable: Variable<V, T> get() = this
    abstract val nbrValues: Int
    abstract val parent: Value
    abstract fun value(value: V): Literal

    val reifiedValue: Value get() = if (optional) this else parent

    /**
     *  If a variable is not mandatory, it will always be set to some value when
     *  the parent model is set.
     */
    abstract val optional: Boolean

    abstract fun valueOf(instance: Instance, index: Int, parentLiteral: Int): T?

    open fun implicitConstraints(
        scope: Scope,
        index: VariableIndex
    ): Sequence<Constraint> = emptySequence()
}

/**
 * This is used for the top variable of the variable hierarchy.
 * It does not take up any space in the optimization problem.
 */
class Root(name: String) : Variable<Nothing, Unit>(name) {
    override val nbrValues get() = 0
    override val optional: Boolean get() = false
    override val parent: Value get() = error("Root does not have a parent.")
    override fun rebase(parent: Value) = error("Root cannot be rebased.")
    override fun valueOf(instance: Instance, index: Int, parentLiteral: Int) {}
    override fun toLiteral(variableIndex: VariableIndex) = error(
        "Root cannot be used in an expression. " +
                "This is likely caused by using a mandatory variable defined " +
                "in the root scope in an expression." )

    override fun value(value: Nothing) =
        error("Root cannot be used as a value.")

    override fun toString() = "Root($name)"
}

/**
 * This is the simplest type of [Variable] that will either be a constant value
 * when the corresponding binary value is 1 or null otherwise.
 */
class Flag<out T>(name: String, val value: T, override val parent: Value) :
    Variable<Nothing, T>(name) {
    override val nbrValues: Int get() = 1
    override val optional: Boolean get() = true
    override fun rebase(parent: Value) = Flag(name, value, parent)
    override fun valueOf(instance: Instance, index: Int, parentLiteral: Int) =
        if (instance.isSet(index)) value else null

    override fun toLiteral(variableIndex: VariableIndex) =
        variableIndex.valueIndexOf(this).toLiteral(true)

    override fun value(value: Nothing) =
        throw UnsupportedOperationException("Cannot be called.")

    override fun toString() = "Flag($name)"
}
