package combo.sat.optimizers

import combo.math.FloatVector
import combo.math.nextNormal
import combo.model.TestModels
import combo.sat.IterationsReachedException
import combo.sat.Problem
import kotlin.test.Test
import kotlin.random.Random
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class Sat4JSolverTest : OptimizerTest() {

    override fun <O : ObjectiveFunction> optimizer(problem: Problem, randomSeed: Int) = null
    override fun <O : ObjectiveFunction> infeasibleOptimizer(problem: Problem, randomSeed: Int) = null

    override fun satOptimizer(problem: Problem, randomSeed: Int) = Sat4JSolver(problem, randomSeed)
    override fun linearOptimizer(problem: Problem, randomSeed: Int) = Sat4JSolver(problem, randomSeed)

    override fun infeasibleSatOptimizer(problem: Problem, randomSeed: Int) = Sat4JSolver(problem, randomSeed)
    override fun infeasibleLinearOptimizer(problem: Problem, randomSeed: Int) = Sat4JSolver(problem, randomSeed)

    override fun timeoutSatOptimizer(problem: Problem, randomSeed: Int) = Sat4JSolver(problem, randomSeed, 0L, maxConflicts = 1)
    override fun timeoutLinearOptimizer(problem: Problem, randomSeed: Int) = Sat4JSolver(problem, randomSeed, 0L, maxConflicts = 1)

    override fun numericSatOptimizer(problem: Problem, randomSeed: Int) = null

    override fun largeSatOptimizer(problem: Problem, randomSeed: Int) = null
    override fun largeLinearOptimizer(problem: Problem, randomSeed: Int) = null

    @Test
    fun maxConflictsTimeout() {
        val p = TestModels.UNSAT_PROBLEMS[2]
        val solver = Sat4JSolver(p, maxConflicts = 1)
        assertFailsWith(IterationsReachedException::class) {
            solver.witnessOrThrow()
        }
        assertEquals(0, solver.asSequence().count())
        assertFailsWith(IterationsReachedException::class) {
            solver.optimizeOrThrow(LinearObjective(true, FloatVector(FloatArray(p.nbrValues) { Random.nextNormal() })))
        }
    }
}
