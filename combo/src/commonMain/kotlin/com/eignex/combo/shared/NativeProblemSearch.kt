package com.eignex.combo.shared

import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.backtrack.BacktrackSolver
import com.eignex.klause.factor.arithmetic.Linear
import com.eignex.klause.factor.bool.Clause
import com.eignex.klause.factor.table.Table
import com.eignex.klause.ir.Factor
import com.eignex.klause.ir.LinearOp
import com.eignex.klause.ir.Lit
import com.eignex.klause.ir.Problem
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.solver.Sample
import com.eignex.klause.solver.SolveResult
import com.eignex.klause.solver.objective.LinearObjective
import com.eignex.klause.solver.result.QuadraticOptimizationResult
import com.eignex.klause.solver.result.SampleResult
import com.eignex.klause.solver.solveSample

/** Source pins are ordinary native rows, so open columns never enter a finite assumptions lane. */
internal fun Problem.pinned(assumptions: Assumptions): Problem {
    val rows = mutableListOf<Factor>()
    fun integer(id: Int, value: Long, op: LinearOp) {
        rows += Linear(longArrayOf(1), intArrayOf(id), op, value)
    }
    assumptions.forEachBool { id, value -> rows += Clause(intArrayOf(Lit.make(id, value))) }
    assumptions.forEachInt { id, value -> integer(id, value, LinearOp.EQ) }
    assumptions.deductions.forEachIntMin { id, value -> integer(id, value, LinearOp.GE) }
    assumptions.deductions.forEachIntMax { id, value -> integer(id, value, LinearOp.LE) }
    assumptions.deductions.forEachIntHole { id, value -> integer(id, value, LinearOp.NE) }
    assumptions.deductions.forEachIntSet { id, values -> rows += Table(intArrayOf(id), values) }
    if (rows.isEmpty()) return this
    return withFactors(factors + rows, impliedFactorMask?.copyOf(factors.size + rows.size))
}

internal fun Problem.sourceMinimize(objective: LinearObjective, params: BacktrackParams): QuadraticOptimizationResult =
    BacktrackSolver.source(pinned(params.assumptions)).optimize(
        objective,
        params.copy(assumptions = Assumptions.None),
    ).result

internal fun Problem.sourceSolve(params: BacktrackParams): SolveResult =
    BacktrackSolver.source(pinned(params.assumptions)).solveSample(params.copy(assumptions = Assumptions.None))

internal fun SolveResult.sampleResult(): SampleResult = when (this) {
    is SolveResult.Sat -> SampleResult.Found(assignment)
    is SolveResult.Unsat -> SampleResult.Infeasible()
    is SolveResult.Unknown -> SampleResult.Unknown(reason)
}

internal val QuadraticOptimizationResult.assignment: Sample?
    get() = (this as? QuadraticOptimizationResult.WithIncumbent)?.incumbent?.sample

internal fun QuadraticOptimizationResult.sampleResult(): SampleResult =
    assignment?.let { SampleResult.Found(it) } ?: when (this) {
        is QuadraticOptimizationResult.Infeasible -> SampleResult.Infeasible()
        is QuadraticOptimizationResult.Unknown -> SampleResult.Unknown(reason)
        else -> error("native result lost its incumbent")
    }
