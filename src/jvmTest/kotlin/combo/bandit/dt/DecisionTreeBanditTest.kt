package combo.bandit.dt

import combo.bandit.PredictionBanditTest
import combo.bandit.TestParameters
import combo.model.Model

class DecisionTreeBanditTest : PredictionBanditTest<DecisionTreeBandit>() {

    override fun bandit(model: Model, parameters: TestParameters) = DecisionTreeBandit.Builder(model, parameters.thompsonPolicy())
            .rewards(parameters.rewards)
            .randomSeed(parameters.randomSeed)
            .maximize(parameters.maximize)
            .maxRestarts(Int.MAX_VALUE)
            .build()

    override fun infeasibleBandit(model: Model, parameters: TestParameters) = DecisionTreeBandit.Builder(model, parameters.thompsonPolicy())
            .rewards(parameters.rewards)
            .randomSeed(parameters.randomSeed)
            .maximize(parameters.maximize)
            .maxRestarts(1)
            .build()
}
