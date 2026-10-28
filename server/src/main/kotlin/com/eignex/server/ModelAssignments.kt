package com.eignex.server

import com.eignex.combo.decisions.pinFloats
import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.backtrack.BacktrackSolver
import com.eignex.combo.decisions.DecisionCoordinates
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.minimizeNative
import com.eignex.klause.solver.objective.LinearObjective
import com.eignex.klause.solver.result.QuadraticOptimizationResult
import com.eignex.klause.util.fitsLong
import com.eignex.klause.model.FloatSpec
import com.eignex.klause.model.SchemaEntry
import com.eignex.klause.propagation.bake
import com.eignex.klause.schema.FloatHandle
import com.eignex.klause.solver.SolveResult
import kotlinx.serialization.json.*

/** A row-only conditioning keeps all coordinate ids and exact values, but changes the solver's frame. */
internal fun com.eignex.klause.solver.Sample.compatibilitySnapshot(): com.eignex.klause.solver.Sample =
    com.eignex.klause.solver.Sample(
        bools, if (exactInts?.any { !it.fitsLong() } == true) longArrayOf() else ints, reals, exactReals = exactReals, exactInts = exactInts,
        algebraicReals = algebraicReals, symbolicReals = symbolicReals,
    )

internal fun realValues(entries: Map<String, SchemaEntry>, input: JsonObject): Map<FloatHandle, Double> =
    input.filterKeys { entries[it] is FloatSpec }.map { (name, element) ->
        val spec = entries.getValue(name) as FloatSpec
        val primitive = element.jsonPrimitive
        val value = requireNotNull(primitive.takeUnless { it.isString }?.doubleOrNull?.takeIf { it.isFinite() }) {
            "'$name' must be a finite number"
        }
        require(value in spec.min..spec.max) { "'$name' outside its bounds" }
        FloatHandle(name, spec.min, spec.max) to value
    }.toMap()

internal fun checkedAssignment(
    compiled: DecisionCoordinates,
    entries: Map<String, SchemaEntry>,
    input: JsonObject,
    params: BacktrackParams,
): SolveResult {
    if (compiled.schema == null) {
        val space = CompiledDecisionSpace.fromProblem(compiled.problem)
        val result = space.minimizeNative(LinearObjective(), readContext(space, input), params)
        return when (result) {
            is QuadraticOptimizationResult.WithIncumbent -> SolveResult.Sat(result.incumbent.sample)
            is QuadraticOptimizationResult.Infeasible -> SolveResult.Unsat()
            is QuadraticOptimizationResult.Unknown -> SolveResult.Unknown(result.reason)
        }
    }
    val problem = compiled.pinFloats(realValues(entries, input))
    val pins = assumptions(compiled, entries, input)
    val result = BacktrackSolver(problem.bake()).solve(params.copy(assumptions = pins))
    return if (result is SolveResult.Sat) {
        val sample = result.assignment
        SolveResult.Sat(sample.compatibilitySnapshot())
    } else result
}
