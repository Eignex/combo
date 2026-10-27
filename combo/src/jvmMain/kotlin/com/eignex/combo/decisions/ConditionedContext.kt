package com.eignex.combo.decisions

import com.eignex.klause.compile.CompiledSchema
import com.eignex.klause.factor.arithmetic.Linear
import com.eignex.klause.ir.LinearOp
import com.eignex.klause.ir.Problem
import com.eignex.klause.model.FloatSpec
import com.eignex.klause.schema.FloatHandle
import kotlin.math.roundToLong

/** A row-only rewrite keeps every coordinate; drop the temporary problem's frame before original decoding. */
internal fun com.eignex.klause.solver.Sample.unframed(): com.eignex.klause.solver.Sample =
    com.eignex.klause.solver.Sample(
        bools, ints, reals, exactReals = exactReals, exactInts = exactInts,
        algebraicReals = algebraicReals, symbolicReals = symbolicReals,
    )

/** Add native real equality rows without changing any solver coordinate or model feature index. */
fun CompiledDecisionSpace.conditionedProblem(context: Context): Problem =
    if (context === Context.Empty) compiled.problem else compiled.pinFloats(
        contextFloats.associate { it.klauseHandle to context[it] },
    )

/** Exact discrete pins for native floats represented on a declared grid. */
fun CompiledSchema.floatAssumptions(values: Map<FloatHandle, Double>): com.eignex.klause.propagation.Assumptions {
    var pins = com.eignex.klause.propagation.Assumptions.None
    for ((handle, value) in values) {
        require(value.isFinite() && value in handle.min..handle.max) { "'${handle.name}' outside its bounds" }
        if (minimize(handle).realCoefficients.any { it != 0.0 }) continue
        intVarIdByName[handle.name]?.let { id ->
            pins = pins.withInt(id, floatBucket(floatDecoders.getValue(handle.name), value))
        }
    }
    return pins
}

/** Solve-time real pins shared by decision contexts and assignment validation. */
fun CompiledSchema.pinFloats(values: Map<FloatHandle, Double>): Problem {
    val rows = values.mapNotNull { (handle, value) ->
        require(value.isFinite() && value in handle.min..handle.max) { "'${handle.name}' outside its bounds" }
        val id = minimize(handle).realCoefficients.indexOfFirst { it != 0.0 }
        if (id < 0) {
            require(handle.name in intVarIdByName) { "'${handle.name}' has no numeric coordinate" }
            return@mapNotNull null
        }
        Linear(intArrayOf(), doubleArrayOf(), intArrayOf(id), doubleArrayOf(1.0), LinearOp.EQ, value)
    }
    return if (rows.isEmpty()) problem else problem.withFactors(problem.factors + rows)
}

internal fun floatBucket(spec: FloatSpec, value: Double): Long {
    require(value.isFinite() && value in spec.min..spec.max) { "float context outside its bounds" }
    val bucket = if (spec.scale == 0.0) 0L else ((value - spec.min) / spec.scale).roundToLong()
    require(bucket in 0L until requireNotNull(spec.buckets).toLong() && spec.realValue(bucket.toInt()) == value) {
        "float context must be exactly representable on its declared grid"
    }
    return bucket
}
