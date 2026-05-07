package combo.expressions

import combo.model.Scope
import combo.model.VariableIndex
import combo.sat.Instance
import combo.sat.toLiteral
import java.util.concurrent.atomic.AtomicInteger

/**
 * This class represents a decision variable in the optimization problem.
 * They must be registered in the [combo.model.Model] to be used.
 * The easiest way of constructing them is through the various methods in
 * [combo.model.Model.Builder], such as [combo.model.Model.Builder.flag] or
 * [combo.model.Model.Builder.nominal] which will also add the required
 * constraints.
 * @param V the type that sub options are parameterized by.
 * @param T the type that is returned, often same as [V].
 */
interface Variable<in V, out T> {

    val name: String
    val nbrValues: Int
    fun value(value: V): Literal

    val optional: Boolean

    fun valueOf(instance: Instance, index: Int, parentLiteral: Int): T?

    fun implicitConstraints(
        scope: Scope,
        index: VariableIndex
    ): Sequence<Constraint> = emptySequence()
}

/**
 * This is used for the top variable of the variable hierarchy.
 * It does not take up any space in the optimization problem.
 */
class Root(override val name: String) : Variable<Nothing, Unit> {
    override val nbrValues get() = 0
    override val optional: Boolean get() = false
    override fun valueOf(instance: Instance, index: Int, parentLiteral: Int) {}
    override fun value(value: Nothing) =
        throw UnsupportedOperationException("Root cannot be used as a value.")

    override fun toString() = "Root($name)"
}

/**
 * This is the simplest type of [Variable] that will either be a constant value
 * when the corresponding binary value is 1 or null otherwise.
 */
class Flag<out T>(override val name: String, val value: T) :
    Value, Variable<Nothing, T> {
    override val nbrValues: Int get() = 1
    override val optional: Boolean get() = true
    override val canonicalVariable: Variable<*, *> get() = this
    override fun valueOf(instance: Instance, index: Int, parentLiteral: Int) =
        if (instance.isSet(index)) value else null

    override fun toLiteral(variableIndex: VariableIndex) =
        variableIndex.valueIndexOf(this).toLiteral(true)

    override fun value(value: Nothing) =
        throw UnsupportedOperationException("Cannot be called.")

    override fun toString() = "Flag($name)"
}
