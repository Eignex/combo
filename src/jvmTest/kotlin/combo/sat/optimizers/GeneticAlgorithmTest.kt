package combo.sat.optimizers

import combo.ga.FixedRateMutation
import combo.ga.PropagatingMutator
import combo.sat.InitializerType
import combo.sat.Problem
import combo.sat.TransitiveImplications

class GeneticAlgorithmTest : OptimizerTest() {
    override fun <O : ObjectiveFunction> optimizer(problem: Problem, randomSeed: Int): Optimizer<O>? {
        return GeneticAlgorithm.Builder(problem).randomSeed(randomSeed).candidateSize(80).penalty(DisjunctPenalty()).timeout(5 * 1000L).build()
    }

    override fun largeLinearOptimizer(problem: Problem, randomSeed: Int) = GeneticAlgorithm.Builder(problem)
            .randomSeed(randomSeed)
            .timeout(5 * 1000L)
            .candidateSize(20)
            .maxSteps(100)
            .stallSteps(10)
            .stallEps(1e-2f)
            .initializer(InitializerType.WEIGHT_MAX_PROPAGATE_COERCE)
            .penalty(DisjunctPenalty())
            .build()

    override fun largeSatOptimizer(problem: Problem, randomSeed: Int) = GeneticAlgorithm.Builder(problem)
            .randomSeed(randomSeed)
            .timeout(5 * 1000L)
            .candidateSize(20)
            .initializer(InitializerType.COERCE)
            .mutation(PropagatingMutator(FixedRateMutation(), TransitiveImplications(problem)))
            .build()

    override fun <O : ObjectiveFunction> infeasibleOptimizer(problem: Problem, randomSeed: Int): Optimizer<O>? {
        return GeneticAlgorithm.Builder(problem).randomSeed(randomSeed).candidateSize(3).timeout(1L).maxSteps(1).restarts(1).build()
    }
}
