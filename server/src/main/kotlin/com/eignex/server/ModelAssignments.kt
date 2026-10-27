package com.eignex.server

import com.eignex.combo.decisions.pinFloats
import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.backtrack.BacktrackSolver
import com.eignex.klause.compile.CompiledSchema
import com.eignex.klause.model.FloatSpec
import com.eignex.klause.model.SchemaEntry
import com.eignex.klause.propagation.bake
import com.eignex.klause.schema.FloatHandle
import com.eignex.klause.solver.SolveResult
import kotlinx.serialization.json.*

/** A row-only conditioning keeps all coordinate ids and exact values, but changes the solver's frame. */
internal fun com.eignex.klause.solver.Sample.compatibilitySnapshot(): com.eignex.klause.solver.Sample =
    com.eignex.klause.solver.Sample(
        bools, ints, reals, exactReals = exactReals, exactInts = exactInts,
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
    compiled: CompiledSchema,
    entries: Map<String, SchemaEntry>,
    input: JsonObject,
    params: BacktrackParams,
): SolveResult {
    val problem = compiled.pinFloats(realValues(entries, input))
    val pins = assumptions(compiled, entries, input)
    val result = BacktrackSolver(problem.bake()).solve(params.copy(assumptions = pins))
    return if (result is SolveResult.Sat) {
        val sample = result.assignment
        SolveResult.Sat(sample.compatibilitySnapshot())
    } else result
}
