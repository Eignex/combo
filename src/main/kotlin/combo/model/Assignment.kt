package combo.model

import combo.expressions.Literal
import combo.expressions.Root
import combo.expressions.Value
import combo.expressions.Variable
import combo.sat.*
import combo.util.IntHashSet

/**
 * This class represents an easy to use way of extracting typed variable values out of an [Instance]. This does not use
 * much additional memory unless it is converted to a [Map] through [toMap]. The tree structure of the [Model] can be
 * used utilized through the [subAssignment] so that sub-models can be properly isolated.
 */
class Assignment(val instance: Instance, val index: VariableIndex, val scope: Scope) : Iterable<Assignment.VariableAssignment<*>> {

    constructor(instance: Instance, index: VariableIndex, scope: Scope, values: Array<out Literal>) : this(instance.also {
        val set = IntHashSet()
        values.forEach {
            if (it is Value) instance.set(it.toLiteral(index))
            else it.collectLiterals(index, set)
            instance.setAll(set)
        }
    }, index, scope)


    fun asSequence(): Sequence<VariableAssignment<*>> = scope.asSequence().mapNotNull {
        val v = it.valueOf(instance, index.valueIndexOf(it), it.parentLiteral(index))
        if (v == null) null
        else VariableAssignment(it, v)
    }

    override fun iterator() = asSequence().iterator()

    fun getString(name: String): String = getOrDefault(name, "")
    fun getChar(name: String): Char = getOrDefault(name, '\u0000')
    fun getBoolean(name: String): Boolean = getOrDefault(name, false)
    fun getLong(name: String): Long = getOrDefault(name, 0L)
    fun getInt(name: String): Int = getOrDefault(name, 0)
    fun getShort(name: String): Short = getOrDefault(name, 0.toShort())
    fun getByte(name: String): Byte = getOrDefault(name, 0.toByte())
    fun getDouble(name: String): Double = getOrDefault(name, 0.0)
    fun getFloat(name: String): Float = getOrDefault(name, 0.0f)

    fun getString(variable: Variable<*, String>): String = getOrDefault(variable, "")
    fun getChar(variable: Variable<*, Char>): Char = getOrDefault(variable, '\u0000')
    fun getBoolean(variable: Variable<*, Boolean>): Boolean = getOrDefault(variable, false)
    fun getLong(variable: Variable<*, Long>): Long = getOrDefault(variable, 0L)
    fun getInt(variable: Variable<*, Int>): Int = getOrDefault(variable, 0)
    fun getShort(variable: Variable<*, Short>): Short = getOrDefault(variable, 0.toShort())
    fun getByte(variable: Variable<*, Byte>): Byte = getOrDefault(variable, 0.toByte())
    fun getDouble(variable: Variable<*, Double>): Double = getOrDefault(variable, 0.0)
    fun getFloat(variable: Variable<*, Float>): Float = getOrDefault(variable, 0.0f)

    operator fun contains(name: String) = scope.find<Variable<*, *>>(name)?.let { contains(it) } ?: false

    operator fun contains(value: Value) = if (value is Root) true else {
        val lit = value.toLiteral(index)
        lit == instance.literal(lit.toIx())
    }
    // = instance.lit variable.valueOf(instance, index.valueIndexOf(variable), variable.parentLiteral(index)) != null

    operator fun <V> get(name: String): V? = scope.find<Variable<*, V>>(name)?.let {
        it.valueOf(instance, index.valueIndexOf(it), it.parentLiteral(index))
    }

    operator fun <V> get(variable: Variable<*, V>): V? = variable.valueOf(instance, index.valueIndexOf(variable), variable.parentLiteral(index))

    fun <V> getOrThrow(variable: Variable<*, V>): V = get(variable)
            ?: throw NoSuchElementException("Variable $variable not found in assignment.")

    fun <V> getOrThrow(name: String): V = get<V>(name)
            ?: throw NoSuchElementException("Variable $name not found in assignment.")

    fun <V> getOrDefault(variable: Variable<*, V>, default: V): V = get(variable) ?: default
    fun <V> getOrDefault(name: String, default: V): V = get(name) ?: default

    fun subAssignment(scopeName: String) = Assignment(instance, index, scope.children.find { it.scopeName == scopeName }
            ?: throw NoSuchElementException("Could not find child scope $scopeName"))

    fun toMap(): Map<Variable<*, *>, Any?> {
        val ret = HashMap<Variable<*, *>, Any?>()
        for (f in scope.asSequence()) {
            val v = get(f)
            if (v != null) ret[f] = v
        }
        return ret
    }

    override fun toString() = asSequence().joinToString(prefix = "{", postfix = "}")
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false
        other as Assignment
        if (instance != other.instance) return false
        return true
    }

    override fun hashCode() = instance.hashCode()

    data class VariableAssignment<V>(val variable: Variable<*, V>, val value: V) {
        override fun toString() = "${variable.name}=$value"
        @Suppress("UNCHECKED_CAST")
        fun toLiteral() = (variable as Variable<Any, V>).value(value as Any)
    }
}
