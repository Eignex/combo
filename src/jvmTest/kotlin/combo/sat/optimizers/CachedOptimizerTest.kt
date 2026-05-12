package combo.sat.optimizers

import combo.sat.InitializerType
import combo.sat.Problem

class CachedOptimizerTest : OptimizerTest() {

    override fun <O : ObjectiveFunction> optimizer(problem: Problem, randomSeed: Int) =
            CachedOptimizer.Builder(LocalSearch.Builder(problem).randomSeed(randomSeed).timeout(5 * 1000L).build())
                    .build()

    override fun linearOptimizer(problem: Problem, randomSeed: Int) =
            CachedOptimizer.Builder(LocalSearch.Builder(problem).initializer(InitializerType.WEIGHT_MAX_PROPAGATE_COERCE)
                    .randomSeed(randomSeed).timeout(5 * 1000L).build())
                    .build()

    override fun <O : ObjectiveFunction> infeasibleOptimizer(problem: Problem, randomSeed: Int) =
            CachedOptimizer.Builder(LocalSearch.Builder(problem).randomSeed(randomSeed).timeout(1L).maxSteps(1).restarts(1).build())
                    .build()
}

