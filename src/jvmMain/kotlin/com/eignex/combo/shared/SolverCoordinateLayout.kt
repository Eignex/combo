package com.eignex.combo.shared

import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.klause.schema.FloatHandle

/** Source-coordinate metadata shared by geometry and model encoders; contains no feature expansion. */
internal class SolverCoordinateLayout(space: CompiledDecisionSpace) {
    val numBoolVars = space.compiled.problem.numBoolVars
    val numIntVars = space.compiled.problem.numIntVars
    val numRealVars = space.compiled.problem.numRealVars
    val boolStart = 0
    val intStart = numBoolVars
    val realStart = numBoolVars + numIntVars
    val size = realStart + numRealVars
    val floatSpecs = space.compiled.floatDecoders.mapNotNull { (name, spec) ->
        space.compiled.intVarIdByName[name]?.let { it to spec }
    }.toMap()
    val realHandles = space.compiled.floatDecoders.mapNotNull { (name, spec) ->
        if (name in space.compiled.intVarIdByName) return@mapNotNull null
        val handle = FloatHandle(name, spec.min, spec.max)
        val id = space.compiled.minimize(handle).realCoefficients.indexOfFirst { it != 0.0 }
        check(id >= 0) { "float '$name' has no solver coordinate" }
        id to handle
    }.toMap()
}
