package com.eignex.combo.decisions

import com.eignex.combo.shared.sourceMinimize
import com.eignex.combo.shared.sourceSolve
import com.eignex.klause.solver.SolveResult
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

/** Find a native exact witness, including symbolic values whose optimization is unsupported. */
fun CompiledDecisionSpace.solveNative(
    context: Context = Context.Empty,
    params: BacktrackParams = BacktrackParams(),
): SolveResult {
    require(coordinates.schema == null) { "use the compiled source strategy for schema problems" }
    val pins = discreteAssumptionsFor(context).mergedWith(params.assumptions)
    return conditionedProblem(context).sourceSolve(params.copy(assumptions = pins))
}
