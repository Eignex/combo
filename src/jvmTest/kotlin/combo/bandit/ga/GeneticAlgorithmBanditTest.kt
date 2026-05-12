package combo.bandit.ga

import combo.bandit.BanditTest
import combo.bandit.TestParameters
import combo.model.Model
import combo.sat.optimizers.LocalSearchTest
import combo.sat.optimizers.ObjectiveFunction
import combo.sat.optimizers.Optimizer

class GeneticAlgorithmBanditTest : BanditTest<GeneticAlgorithmBandit>() {
    override fun bandit(model: Model, parameters: TestParameters) = GeneticAlgorithmBandit.Builder(model.problem, parameters.thompsonPolicy())
            .rewards(parameters.rewards)
            .randomSeed(parameters.randomSeed)
            .maximize(parameters.maximize)
            .candidateSize(50)
            .build()

    @Suppress("UNCHECKED_CAST")
    override fun infeasibleBandit(model: Model, parameters: TestParameters) = GeneticAlgorithmBandit.Builder(model.problem, parameters.thompsonPolicy())
            .optimizer(LocalSearchTest().infeasibleSatOptimizer(model.problem, parameters.randomSeed) as Optimizer<ObjectiveFunction>)
            .rewards(parameters.rewards)
            .randomSeed(parameters.randomSeed)
            .maximize(parameters.maximize)
            .build()
}
