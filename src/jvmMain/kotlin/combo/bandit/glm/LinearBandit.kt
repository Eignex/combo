package combo.bandit.glm

import combo.bandit.ParallelPredictionBandit
import combo.bandit.PredictionBandit
import combo.bandit.PredictionBanditBuilder
import combo.math.DataSample
import combo.math.VoidSample
import combo.model.EffectCodedVector
import combo.model.Model
import combo.sat.BitArray
import combo.sat.Instance
import combo.sat.optimizers.*
import combo.util.IntCollection
import combo.util.RandomSequence

/**
 * Generalized linear model (GLM) bandit with Thompson sampling. The most important configuration parameter is [family],
 * which should be set according to the type of rewards the bandit receives.
 *
 * @param model Model of the search space with variables and constraints.
 * @param linearModel Initial weight data to start with.
 * @param randomSeed Set the random seed to a specific value to have a reproducible algorithm.
 * @param maximize Whether the bandit should maximize or minimize the total rewards. By default true.
 * @param optimizer Which optimizer to use for maximization of the linear function.
 * @param rewards All rewards are added to this for inspecting how well the bandit performs.
 * @param trainAbsError The total absolute error obtained on a prediction before update.
 * @param testAbsError The total absolute error obtained on a prediction after update.
 */
class LinearBandit(val model: Model,
                   val linearModel: LinearModel,
                   override val randomSeed: Int = System.currentTimeMillis().toInt(),
                   override val maximize: Boolean = true,
                   val optimizer: Optimizer<ObjectiveFunction> = LocalSearch.Builder(model.problem).randomSeed(randomSeed).build(),
                   val effectCoding: Boolean = true,
                   override val rewards: DataSample = VoidSample,
                   override val trainAbsError: DataSample = VoidSample,
                   override val testAbsError: DataSample = VoidSample)
    : PredictionBandit<LinearData> {

    private val randomSequence = RandomSequence(randomSeed)

    init {
        require(linearModel.weights.size == model.problem.nbrValues) {
            "Weight prior parameter must be same size as problem: ${model.problem.nbrValues}."
        }
    }

    override fun predict(instance: Instance) = if (effectCoding) linearModel.predict(EffectCodedVector(model, instance))
    else linearModel.predict(instance)

    override fun train(instance: Instance, result: Float, weight: Float) {
        if (effectCoding) linearModel.train(EffectCodedVector(model, instance), result, weight)
        else linearModel.train(instance, result, weight)
    }

    override fun trainAll(instances: Array<Instance>, results: FloatArray, weights: FloatArray?) {
        if (effectCoding) linearModel.trainAll(Array(instances.size) { EffectCodedVector(model, instances[it]) }, results, weights)
        else linearModel.trainAll(instances, results, weights)
    }

    override fun chooseOrThrow(assumptions: IntCollection): Instance {
        if (model.problem.nbrValues == 0) return BitArray(0)
        val rng = randomSequence.next()
        val sampled = linearModel.sample(rng)
        val function = if (effectCoding) EffectCodedObjective(LinearObjective(maximize, sampled), model)
        else DeltaLinearObjective(maximize, sampled)
        return optimizer.optimizeOrThrow(function, assumptions)
    }

    override fun optimalOrThrow(assumptions: IntCollection): Instance {
        val function = if (effectCoding) EffectCodedObjective(LinearObjective(maximize, linearModel.weights), model)
        else DeltaLinearObjective(maximize, linearModel.weights)
        return optimizer.optimizeOrThrow(function, assumptions)
    }

    override fun importData(data: LinearData) {
        val ratio = if (data.step <= 0L) 1f
        else data.step.toFloat() / (linearModel.step + data.step).toFloat()
        linearModel.importData(data, ratio, ratio)
    }

    override fun exportData() = linearModel.exportData()

    class Builder(val model: Model) : PredictionBanditBuilder<LinearData> {

        private var randomSeed: Int = System.currentTimeMillis().toInt()
        private var rewards: DataSample = VoidSample
        private var maximize: Boolean = true
        private var optimizer: Optimizer<ObjectiveFunction>? = null
        private var trainAbsError: DataSample = VoidSample
        private var testAbsError: DataSample = VoidSample
        private var _model: LinearModel? = null
        private var effectCoding: Boolean = true

        val linearModel: LinearModel get() = _model ?: initLinearModel()

        private fun initLinearModel() = DiagonalizedLinearModel.Builder(model.problem).build()

        override fun randomSeed(randomSeed: Int) = apply { this.randomSeed = randomSeed }
        override fun maximize(maximize: Boolean) = apply { this.maximize = maximize }
        override fun rewards(rewards: DataSample) = apply { this.rewards = rewards }
        override fun trainAbsError(trainAbsError: DataSample) = apply { this.trainAbsError = trainAbsError }
        override fun testAbsError(testAbsError: DataSample) = apply { this.testAbsError = testAbsError }
        override fun parallel() = ParallelPredictionBandit.Builder(this)
        override fun importData(data: LinearData) = apply { linearModel.importData(data) }
        fun linearModel(linearModel: LinearModel) = apply { _model = linearModel }

        /** Whether variables should be -1,1 coded (true) or 0,1 coded. */
        fun effectCoding(effectCoding: Boolean) = apply { this.effectCoding = effectCoding }

        /** Which optimizer to use for maximization of the linear function. */
        fun optimizer(optimizer: Optimizer<ObjectiveFunction>) = apply { this.optimizer = optimizer }

        @Suppress("UNCHECKED_CAST")
        override fun suggestOptimizer(optimizer: Optimizer<*>) = optimizer(optimizer as Optimizer<ObjectiveFunction>)

        private fun defaultOptimizer() = optimizer ?: LocalSearch.Builder(model.problem)
                .randomSeed(randomSeed).fallbackCached().build()

        override fun build(): LinearBandit = LinearBandit(model, linearModel, randomSeed, maximize,
                defaultOptimizer(), effectCoding, rewards, trainAbsError, testAbsError)
    }
}
