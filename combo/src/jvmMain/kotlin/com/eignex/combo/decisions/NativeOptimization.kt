package com.eignex.combo.decisions

import com.eignex.combo.shared.sourceMinimize
import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.solver.objective.LinearObjective
import com.eignex.klause.solver.result.QuadraticOptimizationResult

/**
 * Optimize an imported native problem through Klause's source strategy, without finite boxing.
 * Pins become native rows; the result retains exact incumbents, unboundedness and unattained limits.
 * Schema-compiled callers retain their existing solver/objective binding API.
 */
fun CompiledDecisionSpace.minimizeNative(
    objective: LinearObjective,
    context: Context = Context.Empty,
    params: BacktrackParams = BacktrackParams(),
): QuadraticOptimizationResult {
    require(coordinates.schema == null) { "use the compiled schema's bound source strategy for schema objectives" }
    val pins = discreteAssumptionsFor(context).mergedWith(params.assumptions)
    return conditionedProblem(context).sourceMinimize(objective, params.copy(assumptions = pins))
}
