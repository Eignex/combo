package com.eignex.combo.shared

import com.eignex.combo.decisions.CompiledDecisionSpace

/** Source-coordinate metadata shared by geometry and model encoders; contains no feature expansion. */
internal class SolverCoordinateLayout(space: CompiledDecisionSpace) {
    val numBoolVars = space.coordinates.problem.numBoolVars
    val numIntVars = space.coordinates.problem.numIntVars
    val numRealVars = space.coordinates.problem.numRealVars
    val boolStart = 0
    val intStart = numBoolVars
    val realStart = numBoolVars + numIntVars
    val size = realStart + numRealVars
    val floatSpecs = space.coordinates.floatDecoders.mapNotNull { (name, spec) ->
        space.coordinates.intVarIdByName[name]?.let { it to spec }
    }.toMap()
    val realHandles = space.coordinates.realHandles
}
