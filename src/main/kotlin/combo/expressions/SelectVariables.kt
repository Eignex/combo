package combo.expressions

import combo.model.Scope
import combo.model.VariableIndex
import combo.sat.*
import combo.util.IntRangeCollection


/**
 * A [Select] can be either [Nominal], or [Multiple], depending on whether the options in the [values]
 * are mutually exclusive or not. For example, selecting a number of displayed items for a GUI item would be best served
 * as an [Nominal] because there can only a single number at a time.
 */
sealed class Select<V, out T>(
    name: String,
    override val optional: Boolean,
    override val parent: Value,
    values: Array<out V>
) : Variable<V, T>(name) {

    init {
        require(values.isNotEmpty())
    }

    override val nbrValues: Int = values.size + if (optional) 1 else 0
    val values: Array<out Option> =
        Array(values.size) { Option(it, values[it]) }

    override fun value(value: V): Option {
        for (i in values.indices)
            if (values[i].value == value) return values[i]
        throw IllegalArgumentException(
            "Value missing in variable $name. " +
                    "Expected to find $value in ${
                        values.joinToString(
                            prefix = "[",
                            postfix = "]"
                        ) { it.value.toString() }
                    }"
        )
    }

    /**
     * If a specific option in the [Select.values] array need to be used in a constraint, then use this to get a reference
     * to the corresponding optimization variable.
     */
    inner class Option(val valueIndex: Int, val value: V) : Value {

        override val canonicalVariable: Select<V, T> get() = this@Select
        override val name: String get() = canonicalVariable.name

        @Suppress("UNCHECKED_CAST")
        override fun rebase(parent: Value) =
            (parent.canonicalVariable as Select<V, T>).value(value)

        override fun toLiteral(variableIndex: VariableIndex) =
            (variableIndex.valueIndexOf(canonicalVariable) + valueIndex
                    + if (optional) 1 else 0).toLiteral(true)

        override fun toString() = "Option($name=$value)"
    }
}

class Multiple<V>(
    name: String,
    optional: Boolean,
    parent: Value,
    vararg values: V
) : Select<V, List<V>>(name, optional, parent, values) {

    override fun rebase(parent: Value): Variable<*, *> =
        Multiple(name, optional, parent, values)

    override fun valueOf(
        instance: Instance,
        index: Int,
        parentLiteral: Int
    ): List<V>? {
        if ((parentLiteral != 0 && instance.literal(parentLiteral.toIx()) != parentLiteral) || (optional && !instance.isSet(
                index
            ))
        ) return null
        val ret = ArrayList<V>()
        val valueIndex = index + if (optional) 1 else 0
        var i = 0
        while (i < values.size) {
            val value =
                instance.getFirst(valueIndex + i, valueIndex + values.size)
            if (value < 0) break
            i += value
            ret.add(values[i].value)
            i++
        }
        return if (!ret.isEmpty()) ret
        else error("Inconsistent instance, should have something set for $this.")
    }

    override fun implicitConstraints(
        scope: Scope,
        index: VariableIndex
    ): Sequence<Constraint> {
        val firstOption = values[0].toLiteral(index)
        val optionSet =
            IntRangeCollection(firstOption, firstOption + values.size - 1)
        return if (!optional && parent is Root) sequenceOf(Disjunction(optionSet))
        else sequenceOf(
            ReifiedEquivalent(
                reifiedValue.toLiteral(index),
                Disjunction(optionSet)
            )
        )
    }

    override fun toString() = "Multiple($name)"
}

class Nominal<V>(
    name: String,
    optional: Boolean,
    parent: Value,
    vararg values: V
) : Select<V, V>(name, optional, parent, values) {

    override fun rebase(parent: Value): Variable<*, *> =
        Nominal(name, optional, parent, values)

    override fun valueOf(
        instance: Instance,
        index: Int,
        parentLiteral: Int
    ): V? {
        if ((parentLiteral != 0 && instance.literal(parentLiteral.toIx()) != parentLiteral) || (optional && !instance.isSet(
                index
            ))
        ) return null
        val valueIndex = index + if (optional) 1 else 0
        val value = instance.getFirst(valueIndex, valueIndex + values.size)
        return if (value >= 0) values[value].value
        else error("Inconsistent variable, should have something set for $this.")
    }

    override fun implicitConstraints(
        scope: Scope,
        index: VariableIndex
    ): Sequence<Constraint> {
        val firstOption = values[0].toLiteral(index)
        val optionSet =
            IntRangeCollection(firstOption, firstOption + values.size - 1)
        return sequenceOf(
            if (reifiedValue is Root) Disjunction(optionSet)
            else ReifiedEquivalent(
                reifiedValue.toLiteral(index),
                Disjunction(optionSet)
            ),
            Cardinality(optionSet, 1, Relation.LE)
        )
    }

    override fun toString() = "Nominal($name)"
}
