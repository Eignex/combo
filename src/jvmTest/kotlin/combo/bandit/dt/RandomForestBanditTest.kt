package combo.bandit.dt

import combo.bandit.BanditTest
import combo.bandit.TestParameters
import combo.model.Model

class RandomForestBanditTest : BanditTest<RandomForestBandit>() {
    override fun bandit(model: Model, parameters: TestParameters) = RandomForestBandit.Builder(model, parameters.thompsonPolicy())
            .rewards(parameters.rewards)
            .randomSeed(parameters.randomSeed)
            .maximize(parameters.maximize)
            .build()
}
