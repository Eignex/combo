package com.eignex.server

import com.eignex.klause.compile.CompiledSchema
import com.eignex.klause.model.FloatSpec
import com.eignex.klause.model.SchemaEntry
import com.eignex.klause.schema.FloatHandle
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.Sample
import com.eignex.klause.solver.SolveResult
import kotlinx.serialization.json.*
import kotlin.math.abs
import kotlin.math.roundToLong

internal fun checkedAssignment(
    compiled: CompiledSchema,
    entries: Map<String, SchemaEntry>,
    input: JsonObject,
    solve: (com.eignex.klause.propagation.Assumptions) -> SolveResult,
): SolveResult {
    val floats = input.filterKeys { entries[it] is FloatSpec }
    val realValues = floats.mapValues { (name, value) ->
        val spec = entries.getValue(name) as FloatSpec
        val primitive = value.jsonPrimitive
        requireNotNull(primitive.takeUnless { it.isString }?.doubleOrNull?.takeIf {
            it.isFinite() && it in spec.min..spec.max
        }) { "'$name' must be a number within its bounds" }
    }
    val pins = assumptions(compiled, entries, JsonObject(input.filterKeys { it !in floats }))
    val result = solve(pins)
    if (result !is SolveResult.Sat || floats.isEmpty()) return result
    val anchor = result.assignment
    val reals = anchor.reals
    val exact = (anchor.exactReals ?: reals.map { BigFraction.ofDouble(it)!! }).toMutableList()
    val integers = anchor.ints
    for ((name, value) in realValues) {
        val spec = entries.getValue(name) as FloatSpec
        val coefficients = compiled.minimize(FloatHandle(name, spec.min, spec.max)).realCoefficients
        val id = coefficients.indexOfFirst { it != 0.0 }
        if (id >= 0) {
            reals[id] = value
            exact[id] = BigFraction.ofDouble(value)!!
        } else {
            val grid = compiled.floatDecoders.getValue(name)
            val bucket = if (grid.scale == 0.0) 0L else ((value - grid.min) / grid.scale).roundToLong()
            require(abs(grid.realValue(bucket.toInt()) - value) <= 1e-10) { "'$name' is outside the supported grid" }
            integers[compiled.intVarIdByName.getValue(name)] = bucket
        }
    }
    return SolveResult.Sat(Sample(anchor.bools, integers, reals, exactReals = exact))
}
