package combo.sat.optimizers

import combo.sat.Problem

class ExhaustiveSolverTest : OptimizerTest() {
    override fun <O : ObjectiveFunction> optimizer(problem: Problem, randomSeed: Int) = ExhaustiveSolver(problem, randomSeed)
    override fun <O : ObjectiveFunction> infeasibleOptimizer(problem: Problem, randomSeed: Int) = ExhaustiveSolver(problem, randomSeed, 1L)
    override fun largeLinearOptimizer(problem: Problem, randomSeed: Int) = null
    override fun largeSatOptimizer(problem: Problem, randomSeed: Int) = null
    override fun numericSatOptimizer(problem: Problem, randomSeed: Int) = null
}
