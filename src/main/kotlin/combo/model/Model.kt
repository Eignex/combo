package combo.model

import combo.sat.*
import combo.sat.constraints.Conjunction
import combo.sat.constraints.Relation
import combo.util.*
import kotlin.jvm.JvmOverloads

/**
 * The model consists of a constraint satisfaction [Problem] and variables defined in a [Scope]. A [VariableIndex]
 * provides a mapping between the variables defined in the scope and the numeric indices they have in constraints.
 */
class Model(val problem: Problem, val index: VariableIndex, val scope: Scope) {

    val nbrVariables: Int get() = index.nbrVariables

    val reifiedLiterals: IntArray = IntArray(problem.nbrValues)
    // TODO could be a range with binary search
    // like so : private data class ReifiedLiteral(val index: Int, val range: IntRange)

    init {
        for (variable in index) {
            val ix = index.valueIndexOf(variable)
            val offset = if (variable.optional) {
                reifiedLiterals[ix] = variable.parentLiteral(index)
                1
            } else {
                0
            }
            val valueReification = if (variable.reifiedValue is Root) 0
            else variable.reifiedValue.toLiteral(index)
            if (valueReification != 0)
                for (i in offset until variable.nbrValues)
                    reifiedLiterals[ix + i] = valueReification
        }
    }

    /**
     * Create an assignment based on setting the literals provided.
     */
    fun denseAssignment(vararg literals: Literal) = Assignment(BitArray(problem.nbrValues), index, scope, literals)

    /**
     * Create a sparse assignment based on setting the literals provided.
     */
    fun sparseAssignment(vararg literals: Literal) = Assignment(SparseBitArray(problem.nbrValues), index, scope, literals)

    /**
     * Wrap [Instance] to [Assignment].
     */
    fun toAssignment(instance: Instance) = Assignment(instance, index, scope)

    /**
     * Wrap [Sequence] of [Instance] to [Sequence] of [Assignment].
     */
    fun toAssignments(sequence: Sequence<Instance>) = sequence.map { toAssignment(it) }

    /**
     * Finds a variable using breadth first search.
     */
    operator fun get(name: String): Variable<*, *> = scope[name]

    /**
     * Finds a variable using breadth first search, with the [value] specified.
     */
    operator fun <V> get(name: String, value: V) = scope[name, value]

    companion object {
        inline fun model(name: String = Variable.defaultName(), init: Builder.() -> Unit): Model {
            val builder = Builder(name)
            init.invoke(builder)
            return builder.build()
        }
    }

    class Builder private constructor(override val index: VariableIndex,
                                      override val scope: RootScope,
                                      override val constraints: MutableList<Constraint>,
                                      override val units: IntHashSet) : ModelBuilder<RootScope>(), Value by scope.reifiedValue {

        @JvmOverloads
        constructor(name: String = Variable.defaultName()) : this(VariableIndex(), RootScope(Root(name)), ArrayList(), IntHashSet())

        override fun toString() = "Builder($name)"
    }

    @ModelMarker
    abstract class ModelBuilder<S : Scope> : Value {

        abstract val index: VariableIndex
        abstract val scope: S
        protected abstract val constraints: List<Constraint>
        protected abstract val units: IntCollection

        fun build(): Model {
            val constraints = let {
                val problem = Problem(index.nbrValues, constraints.toTypedArray())
                val reduced = problem.unitPropagation(units as IntHashSet, true)
                if (units.size > 0) reduced + Conjunction(collectionOf(*units.toArray()))
                else reduced
            }
            return Model(Problem(index.nbrValues, constraints), index, scope)
        }

        fun bool(name: String = Variable.defaultName()) = flag(name, true)

        /**
         * Wraps an object which is defined when the underlying variable is true.
         */
        fun <V> flag(name: String = Variable.defaultName(), value: V) =
                Flag(name, value, scope.reifiedValue).apply {
                    addVariable(this)
                }

        /**
         * Only one of the provided values can be defined, with an indicator variable.
         */
        fun <V> optionalNominal(name: String = Variable.defaultName(), vararg values: V) =
                Nominal(name, true, scope.reifiedValue, *values).also {
                    addVariable(it)
                }

        /**
         * Only one of the provided values can be defined.
         */
        fun <V> nominal(name: String = Variable.defaultName(), vararg values: V) =
                Nominal(name, false, scope.reifiedValue, *values).also {
                    addVariable(it)
                }

        /**
         * Any of the provided values can be defined in a [List].
         */
        fun <V> multiple(name: String = Variable.defaultName(), vararg values: V) =
                Multiple(name, false, scope.reifiedValue, *values).also {
                    addVariable(it)
                }

        /**
         * Any of the provided values can be defined in a [List], with an indicator variable.
         */
        fun <V> optionalMultiple(name: String = Variable.defaultName(), vararg values: V) =
                Multiple(name, true, scope.reifiedValue, *values).also {
                    addVariable(it)
                }

        /**
         * Integer value with specified bounds, with an indicator variable.
         */
        fun optionalInt(name: String = Variable.defaultName(), min: Int = Int.MIN_VALUE, max: Int = Int.MAX_VALUE) =
                IntVar(name, true, scope.reifiedValue, min, max).also {
                    addVariable(it)
                }

        /**
         * Integer value with specified bounds.
         */
        fun int(name: String = Variable.defaultName(), min: Int = Int.MIN_VALUE, max: Int = Int.MAX_VALUE) =
                IntVar(name, false, scope.reifiedValue, min, max).also {
                    addVariable(it)
                }

        /**
         * Float value with specified bounds, with an indicator variable.
         */
        fun optionalFloat(name: String = Variable.defaultName(), min: Float = -MAX_VALUE32, max: Float = MAX_VALUE32) =
                FloatVar(name, true, scope.reifiedValue, min, max).also {
                    addVariable(it)
                }

        /**
         * Float value with specified bounds.
         */
        fun float(name: String = Variable.defaultName(), min: Float = -MAX_VALUE32, max: Float = MAX_VALUE32) =
                FloatVar(name, false, scope.reifiedValue, min, max).also {
                    addVariable(it)
                }

        /**
         * Bit field value.
         * @param nbrBits number of bits in the bit field.
         */
        fun bits(name: String = Variable.defaultName(), nbrBits: Int) =
                BitsVar(name, false, scope.reifiedValue, nbrBits).also {
                    addVariable(it)
                }

        /**
         * Bit field value, with an indicator variable.
         * @param nbrBits number of bits in the bit field.
         */
        fun optionalBits(name: String = Variable.defaultName(), nbrBits: Int) =
                BitsVar(name, true, scope.reifiedValue, nbrBits).also {
                    addVariable(it)
                }

        /**
         * Adds an optional sub-model. Every variable in the new model will be controlled by a [bool] named [name]. The
         * scope name will also be [name].
         */
        fun model(name: String = Variable.defaultName(), init: ModelBuilder<ChildScope<S>>.() -> Unit) = model(bool(name), name, init)

        /**
         * A sub model allows organizing groups of variables in a logical hierarchy. Every variable 'v' under this will
         * have the constraint that v => reifiedValue. In this way, variables below this will only be true if the
         * reifiedValue is true.
         * @param reifiedValue value that serves as indicator variable for all variables defined in sub model.
         * @param scopeName name of the scope, used to create [Assignment] on the sub model only.
         */
        fun model(reifiedValue: Value, scopeName: String = reifiedValue.name, init: ModelBuilder<ChildScope<S>>.() -> Unit): ModelBuilder<ChildScope<S>> {
            require(reifiedValue.canonicalVariable == canonicalVariable || index.contains(reifiedValue.canonicalVariable)) { "${reifiedValue.canonicalVariable} not found in model." }
            @Suppress("UNCHECKED_CAST")
            val builder = ChildBuilder(this, scope.addScope(scopeName, reifiedValue) as ChildScope<S>)
            val parentValue = this@ModelBuilder.scope.reifiedValue
            if (!scope.inScope(reifiedValue.canonicalVariable.name) && parentValue !is Root)
                impose {
                    reifiedValue implies parentValue
                }
            init.invoke(builder)
            return builder
        }

        /**
         * Short-hand for adding a child model with the same reified value as the current model. As such,
         * it is only used for lexical scoping.
         * @param scopeName name of the scope, used to create [Assignment] on the sub model only.
         */
        fun scope(scopeName: String = Variable.defaultName(), init: ModelBuilder<ChildScope<S>>.() -> Unit) =
                model(scope.reifiedValue, scopeName, init)

        /**
         * Add a constraint through [ConstraintFactory].
         */
        inline fun impose(init: ConstraintFactory<S>.() -> Expression) = apply {
            val builder = ConstraintFactory(scope, index)
            val exp = init.invoke(builder)
            addConstraint(exp)
        }

        /**
         * Add a new [Variable] to the model. It will also add the implicit constraints from the hierarchy and the
         * variable itself.
         */
        fun addVariable(variable: Variable<*, *>) = apply {
            require(variable.nbrValues > 0)
            index.add(variable)
            scope.add(variable)
            val parentValue = this@ModelBuilder.scope.reifiedValue
            if (variable.optional && parentValue !is Root)
                impose { variable implies parentValue }
            variable.implicitConstraints(scope, index).forEach {
                addConstraint(it)
            }
        }


        /**
         * Adds a separate [Model] as a child model to this. Variables are copied to new model. Constraints in [model]
         * are ignored.
         */
        fun addModel(model: Model) = apply {

            val r = model.scope.reifiedValue as? Root ?: error("Sub model must have Root.")
            val rebased = HashMap<Variable<*, *>, Value>()
            rebased[r] = bool(r.name)
            fun rebaseValue(value: Value): Value {
                return if (value is Variable<*, *>)
                    rebased.getOrPut(value) {
                        value.rebase(rebaseValue(value.parent))
                    } else value.rebase(rebaseValue(value.canonicalVariable.parent))
            }

            val scopes = ArrayList<Scope>()
            val definedIn = ArrayList<ModelBuilder<*>>()
            scopes.add(model.scope)
            definedIn.add(this)
            while (scopes.isNotEmpty()) {
                val next = scopes.removeAt(0)
                val parent = definedIn.removeAt(0)
                /*val rv = if (next.reifiedValue !in index) {
                    when {
                        next.reifiedValue is Root -> parent.bool(next.reifiedValue.name)
                        next.reifiedValue !is Variable<*, *> -> error("Value ${next.reifiedValue} not contained in scope.")
                        else -> parent.addVariable(next.reifiedValue as Variable<*, *>)
                    }
                } else next.reifiedValue*/
                parent.model(rebaseValue(next.reifiedValue), next.scopeName) {
                    next.variables.forEach { addVariable(rebaseValue(it) as Variable<*, *>) }
                    next.children.forEach {
                        scopes.add(it)
                        definedIn.add(this)
                    }
                }
            }
        }

        /**
         * Add a constraint to the model.
         * @param unitPropagation perform unit propagation simplification on the constraint.
         */
        fun addConstraint(expression: Expression, unitPropagation: Boolean = true) = apply {

            fun imposeConstraint(constraint: Constraint) {
                var prop = constraint
                if (unitPropagation) {
                    units.forEach {
                        prop = prop.unitPropagation(it)
                        when (prop) {
                            is Tautology -> return
                            is Empty -> throw UnsatisfiableException("Unsatisfiable by unit propagation.")
                        }
                    }
                }
                if (prop.isUnit())
                    (units as IntHashSet).addAll(prop.unitLiterals())
                (constraints as MutableList).add(prop)
            }

            when (expression) {
                is Literal -> {
                    val set = IntHashSet()
                    expression.collectLiterals(index, set)
                    imposeConstraint(Conjunction(collectionOf(*set.toArray())))
                }
                is CNF -> expression.disjunctions.forEach { imposeConstraint(it) }
                is Constraint -> imposeConstraint(expression)
                else -> throw UnsupportedOperationException(
                        "Custom types of expression cannot be added as constraints in this way.")
            }
        }

        /**
         * Add something with fluid syntax.
         */
        fun add() = SubBuilder(this)

        /**
         * Performs a breadth first search from the current scope and down, throwing an error in case of failure.
         */
        operator fun get(name: String): Variable<*, *> = scope.find(name)
                ?: throw NoSuchElementException("Scope contains no variable with name $name.")

        /**
         * Performs a breadth first search from the current scope and down, throwing an error in case of failure.
         * Then the [value] is extracted from the value.
         */
        operator fun <V> get(name: String, value: V) = scope.find<Variable<V, *>>(name)?.value(value)
                ?: throw NoSuchElementException("Scope contains no variable with name $name.")

        private class ChildBuilder<S : Scope>(val parent: ModelBuilder<*>, override val scope: S) : ModelBuilder<S>(), Value by scope.reifiedValue {
            override val index: VariableIndex get() = parent.index
            override val constraints: List<Constraint> get() = parent.constraints
            override val units: IntCollection get() = parent.units

            override fun toString() = "ChildBuilder($name)"
        }

    }

    /**
     * For java inter-op with fluent builder style.
     */
    class SubBuilder<S : Scope>(val builder: ModelBuilder<S>) {

        /**
         * Wraps an object which is defined when the underlying variable is true.
         */
        @JvmOverloads
        fun <V> flag(name: String = Variable.defaultName(), value: V) = builder.apply { flag(name, value) }

        @JvmOverloads
        fun bool(name: String = Variable.defaultName()) = builder.apply { bool(name) }

        /**
         * Only one of the provided values can be defined, with an indicator variable.
         */
        @JvmOverloads
        fun <V> optionalNominal(name: String = Variable.defaultName(), vararg values: V) = builder.apply { optionalNominal(name, *values) }

        /**
         * Only one of the provided values can be defined.
         */
        @JvmOverloads
        fun <V> nominal(name: String = Variable.defaultName(), vararg values: V) = builder.apply { nominal(name, *values) }

        /**
         * Any of the provided values can be defined in a [List].
         */
        @JvmOverloads
        fun <V> multiple(name: String = Variable.defaultName(), vararg values: V) = builder.apply { multiple(name, *values) }

        /**
         * Any of the provided values can be defined in a [List], with an indicator variable.
         */
        @JvmOverloads
        fun <V> optionalMultiple(name: String = Variable.defaultName(), vararg values: V) = builder.apply { optionalMultiple(name, *values) }

        /**
         * Integer value with specified bounds, with an indicator variable.
         */
        @JvmOverloads
        fun optionalInt(name: String = Variable.defaultName(), min: Int = Int.MIN_VALUE, max: Int = Int.MAX_VALUE) = builder.apply { optionalInt(name, min, max) }

        /**
         * Integer value with specified bounds.
         */
        @JvmOverloads
        fun int(name: String = Variable.defaultName(), min: Int = Int.MIN_VALUE, max: Int = Int.MAX_VALUE) = builder.apply { int(name, min, max) }

        /**
         * Float value with specified bounds, with an indicator variable.
         */
        @JvmOverloads
        fun optionalFloat(name: String = Variable.defaultName(), min: Float = -MAX_VALUE32, max: Float = MAX_VALUE32) = builder.apply { optionalFloat(name, min, max) }

        /**
         * Float value with specified bounds.
         */
        @JvmOverloads
        fun float(name: String = Variable.defaultName(), min: Float = -MAX_VALUE32, max: Float = MAX_VALUE32) = builder.apply { float(name, min, max) }

        /**
         * Bit field value.
         * @param nbrBits number of bits in the bit field.
         */
        @JvmOverloads
        fun bits(name: String = Variable.defaultName(), nbrBits: Int) = builder.apply { bits(name, nbrBits) }

        /**
         * Bit field value, with an indicator variable.
         * @param nbrBits number of bits in the bit field.
         */
        @JvmOverloads
        fun optionalBits(name: String = Variable.defaultName(), nbrBits: Int) = builder.apply { optionalBits(name, nbrBits) }

        /**
         * Any of the variable must be true, logical or.
         */
        fun disjunction(vararg references: String) =
                builder.apply { impose { disjunction(*references.mapArray { builder[it] }) } }

        /**
         * All of the variable must be true, logical and.
         */
        fun conjunction(vararg references: String) =
                builder.apply { impose { conjunction(*references.mapArray { builder[it] }) } }

        /**
         * Specify a relation with weights that are multiplied with variables.
         * x1*w1 + x2*w2 ... + xn*wn [relation] [degree],
         */
        fun linear(degree: Int, relation: Relation, weights: IntArray, vararg references: String) =
                builder.apply { impose { linear(degree, relation, weights, references.mapArray { builder[it] }) } }

        /**
         * Specify a relation between the number of variables that are true.
         */
        fun cardinality(degree: Int, relation: Relation, vararg references: String) =
                builder.apply { impose { cardinality(degree, relation, *references.mapArray { builder[it] }) } }

        /**
         * Precisely this number of variables must be true.
         */
        fun exactly(degree: Int, vararg references: String) =
                builder.apply { impose { exactly(degree, *references.mapArray { builder[it] }) } }

        /**
         * At most this number of variables must be true (defined with less than equal).
         */
        fun atMost(degree: Int, vararg references: String) =
                builder.apply { impose { atMost(degree, *references.mapArray { builder[it] }) } }

        /**
         * At least this number of variables must be true (defined with greater than equal).
         */
        fun atLeast(degree: Int, vararg references: String) =
                builder.apply { impose { atLeast(degree, *references.mapArray { builder[it] }) } }

        /**
         * Declares variables to be mutually exclusive.
         */
        fun excludes(vararg references: String) =
                builder.apply { builder.impose { excludes(*references.mapArray { builder[it] }) } }
    }

    override fun toString() = "Model(${scope.scopeName})"
}

@DslMarker
annotation class ModelMarker
