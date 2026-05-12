package combo.bandit

import combo.model.Model

class ListBanditTest : BanditTest<ListBandit>() {
    override fun bandit(model: Model, parameters: TestParameters) =
            ListBandit.Builder(model.problem, parameters.ucbPolicy())
                    .rewards(parameters.rewards)
                    .randomSeed(parameters.randomSeed)
                    .maximize(parameters.maximize)
                    .build()
}
