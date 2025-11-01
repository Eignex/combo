package combo.model


import combo.expressions.Literal
import combo.sat.ValidationException
import combo.sat.optimizers.*
import combo.util.EmptyCollection
import combo.util.IntCollection
import combo.util.IntHashSet
import kotlin.jvm.JvmStatic

class ModelOptimizer<O : ObjectiveFunction>(val model: Model, val optimizer: Optimizer<O>) {

    companion object {
        @JvmStatic
        fun exhaustive(model: Model) = ModelOptimizer(model, ExhaustiveSolver(model.problem))

        @JvmStatic
        fun localSearch(model: Model) = ModelOptimizer(model, LocalSearch(model.problem))

        @JvmStatic
        fun geneticAlgorithm(model: Model) = ModelOptimizer(model, GeneticAlgorithm(model.problem))
    }

    /**
     * Generates a random solution, ie. a witness.
     * @param assumptions these variables will be fixed during solving, see [combo.expressions.Literal].
     * @param guess starting point for search if one is provided. This instance will be reused if applicable.
     * @throws ValidationException if there is a logical error in the problem or a solution cannot be found with the
     * allotted resources..
     */
    fun witnessOrThrow(vararg assumptions: Literal) =
            model.toAssignment(optimizer.witnessOrThrow(assumptionsLiterals(assumptions)))

    /**
     * Generates a random solution, ie. a witness.
     * @param assumptions these variables will be fixed during solving, see [Literal].
     * @param guess starting point for search if one is provided. This instance will be reused if applicable.
     */
    fun witness(vararg assumptions: Literal): Assignment? {
        val instance = optimizer.witness(assumptionsLiterals(assumptions))
        return if (instance != null) model.toAssignment(instance)
        else null
    }

    fun asSequence(vararg assumptions: Literal) =
            model.toAssignments(optimizer.asSequence(assumptionsLiterals(assumptions)))

    /**
     * Minimize the [function], optionally with the additional constraints in [assumptions].
     * Returns null if no instance can be found.
     * @param function the objective function to optimize on.
     * @param assumptions these variables will be fixed during solving, see [Literal].
     */
    fun optimize(function: O, vararg assumptions: Literal): Assignment? {
        val instance = optimizer.optimize(function, assumptionsLiterals(assumptions))
        return if (instance != null) model.toAssignment(instance)
        else null
    }

    /**
     * Minimize the [function], optionally with the additional constraints in [assumptions].
     * @param function the objective function to optimize on.
     * @param assumptions these variables will be fixed during solving, see [Literal].
     * @throws ValidationException if there is a logical error in the problem or a solution cannot be found with the
     * allotted resources.
     */
    fun optimizeOrThrow(function: O, vararg assumptions: Literal) =
            model.toAssignment(optimizer.optimizeOrThrow(function, assumptionsLiterals(assumptions)))

    private fun assumptionsLiterals(assumptions: Array<out Literal>): IntCollection {
        if (assumptions.isEmpty()) return EmptyCollection
        val set = IntHashSet()
        assumptions.forEach { it.collectLiterals(model.index, set) }
        return set
    }
}
