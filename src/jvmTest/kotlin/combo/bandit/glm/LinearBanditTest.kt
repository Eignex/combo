package combo.bandit.glm

import combo.bandit.BanditTest
import combo.bandit.TestParameters
import combo.model.Model
import combo.sat.optimizers.LocalSearch

class GreedyLinearBanditTest : BanditTest<LinearBandit>() {
    override fun bandit(model: Model, parameters: TestParameters) =
            LinearBandit.Builder(model)
                    .linearModel(SGDLinearModel.Builder(model.problem)
                            .link(parameters.variance().canonicalLink()).build())
                    .randomSeed(parameters.randomSeed)
                    .maximize(parameters.maximize)
                    .rewards(parameters.rewards)
                    .optimizer(LocalSearch.Builder(model.problem).randomSeed(parameters.randomSeed)
                            .maxConsideration(100)
                            .maxSteps(50)
                            .restarts(1)
                            .fallbackCached().build())
            .build()
}

class PrecisionLinearBanditTest : BanditTest<LinearBandit>() {
    override fun bandit(model: Model, parameters: TestParameters) =
            LinearBandit.Builder(model)
                    .linearModel(DiagonalizedLinearModel.Builder(model.problem)
                            .family(parameters.variance()).build())
                    .randomSeed(parameters.randomSeed)
                    .maximize(parameters.maximize)
                    .rewards(parameters.rewards)
                    .optimizer(LocalSearch.Builder(model.problem).randomSeed(parameters.randomSeed)
                            .maxConsideration(100)
                            .maxSteps(50)
                            .restarts(1)
                            .fallbackCached().build())
                    .build()
}

class CovarianceLinearBanditTest : BanditTest<LinearBandit>() {
    override fun bandit(model: Model, parameters: TestParameters) =
            LinearBandit.Builder(model)
                    .linearModel(CovarianceLinearModel.Builder(model.problem)
                            .family(parameters.variance()).build())
                    .randomSeed(parameters.randomSeed)
                    .maximize(parameters.maximize)
                    .rewards(parameters.rewards)
                    .optimizer(LocalSearch.Builder(model.problem).randomSeed(parameters.randomSeed)
                            .maxConsideration(100)
                            .maxSteps(50)
                            .restarts(1)
                            .fallbackCached().build())
                    .build()
}
