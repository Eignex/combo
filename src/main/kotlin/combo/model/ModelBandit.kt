package combo.model

import combo.bandit.Bandit
import combo.bandit.ListBandit
import combo.bandit.PredictionBandit
import combo.bandit.dt.DecisionTreeBandit
import combo.bandit.dt.RandomForestBandit
import combo.bandit.ga.GeneticAlgorithmBandit
import combo.bandit.glm.CovarianceLinearModel
import combo.bandit.glm.LinearBandit
import combo.bandit.glm.VarianceFunction
import combo.bandit.univariate.BanditPolicy
import combo.expressions.Literal
import combo.math.DataSample
import combo.util.EmptyCollection
import combo.util.IntCollection
import combo.util.IntHashSet
import kotlin.jvm.JvmOverloads
import kotlin.jvm.JvmStatic

open class ModelBandit<B : Bandit<*>>(val model: Model, open val bandit: B) {

    companion object {
        @JvmStatic
        fun listBandit(model: Model, banditPolicy: BanditPolicy) =
                ModelBandit(model, ListBandit.Builder(model.problem, banditPolicy).build())

        @JvmStatic
        fun decisionTreeBandit(model: Model, banditPolicy: BanditPolicy) =
                PredictionModelBandit(model, DecisionTreeBandit.Builder(model, banditPolicy).build())

        @JvmStatic
        @JvmOverloads
        fun randomForestBandit(model: Model, banditPolicy: BanditPolicy, nbrTrees: Int = 10) =
                PredictionModelBandit(model, RandomForestBandit.Builder(model, banditPolicy).trees(nbrTrees).build())

        @JvmStatic
        fun geneticAlgorithmBandit(model: Model, banditPolicy: BanditPolicy) =
                ModelBandit(model, GeneticAlgorithmBandit.Builder(model.problem, banditPolicy).build())

        @JvmStatic
        fun linearBandit(model: Model, family: VarianceFunction) =
                PredictionModelBandit(model, LinearBandit.Builder(model).linearModel(
                        CovarianceLinearModel.Builder(model.problem).family(family).build()
                ).build())
    }

    fun choose(vararg assumptions: Literal): Assignment? {
        val instance = bandit.choose(assumptionsLiterals(assumptions))
        return if (instance != null) model.toAssignment(instance)
        else null
    }

    fun chooseOrThrow(vararg assumptions: Literal) =
            model.toAssignment(bandit.chooseOrThrow(assumptionsLiterals(assumptions)))

    fun optimal(vararg assumptions: Literal): Assignment? {
        val instance = bandit.optimal(assumptionsLiterals(assumptions))
        return if (instance != null) model.toAssignment(instance)
        else null
    }

    fun optimalOrThrow(vararg assumptions: Literal) =
            model.toAssignment(bandit.optimalOrThrow(assumptionsLiterals(assumptions)))

    /**
     * Update the result of an assignment.
     *
     * @param assignment the assigned values used for the result.
     * @param result the reward obtained. If this constitutes multiple rewards then set weight appropriately and divide
     * by the [weight].
     * @param weight update strength. Can be used to signal importance of a result. The higher value the more the
     * algorithm is updated. If these are number of trials (or observations) in for example a binomial distributed
     * reward, then the result should be divided by weight before calling update (ie. the [result] should be mean).
     */
    @JvmOverloads
    fun update(assignment: Assignment, result: Float, weight: Float = 1.0f) {
        bandit.update(assignment.instance, result, weight)
    }

    private fun assumptionsLiterals(assumptions: Array<out Literal>): IntCollection {
        if (assumptions.isEmpty()) return EmptyCollection
        val set = IntHashSet()
        assumptions.forEach { it.collectLiterals(model.index, set) }
        return set
    }
}

class PredictionModelBandit<B : PredictionBandit<*>>(model: Model, override val bandit: B) : ModelBandit<B>(model, bandit) {
    /**
     * The total absolute error obtained on a prediction before update.
     */
    var trainAbsError: DataSample = bandit.trainAbsError

    /**
     * The total absolute error obtained on a prediction after update.
     */
    var testAbsError: DataSample = bandit.testAbsError

    /**
     * Evaluate the machine learning model on a [Assignment].
     */
    fun predict(assignment: Assignment): Float = bandit.predict(assignment.instance)
}
