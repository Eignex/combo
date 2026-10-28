package com.eignex.combo.bo

import com.eignex.combo.bo.model.BatchForestConfig
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.test.TimedTest
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import kotlin.math.exp
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs

class ResponseScaleTest : TimedTest() {
    private class Rate : DecisionSpace() {
        val x by floatVar(0.0, 1.0)
    }
    private fun model(): ConditionableSurrogate = object : ConditionableSurrogate {
        override val uncertaintyKind = UncertaintyKind.LatentPosterior
        override fun predict(features: DenseMatrix) = PredictiveMoments(
            DenseVector.zero(features.rows),
            DenseVector.of(DoubleArray(features.rows) { 1.0 }),
        )
        override fun drawFunction(random: Random) = SampledFunction { DenseVector.zero(it.rows) }
        override fun condition(temporary: TrainingData): FittedSurrogate = this
    }

    @Test
    fun `log response EI integrates original unit improvement rather than Gaussian raw moment EI`() {
        val fitted = ResponseTransform(0.0, 1.0, logarithmic = true).model(model())
        val queries = DenseMatrix.ofRows(arrayOf(doubleArrayOf(0.0)))
        assertEquals(exp(0.5), fitted.predict(queries).means[0], 1e-12)
        val acquisition = ExpectedImprovement().create(
            AcquisitionContext(fitted, 1.0, OutcomeSpec(maximize = false)),
            Random(1),
        )
        assertEquals(0.2384217081, acquisition.evaluate(queries)[0], 1e-6)
        assertEquals(1.0, fitted.drawFunction(Random(1)).evaluate(queries)[0])
    }

    @Test
    fun `standardized log models preserve raw log moments and condition transformed noise consistently`() {
        var response = Double.NaN
        var variance = Double.NaN
        val fitted = object : ConditionableSurrogate by model() {
            override fun condition(temporary: TrainingData): FittedSurrogate {
                response = temporary.responses[0]
                variance = temporary.noiseVariances!![0]
                return model()
            }
        }
        val wrapped = ResponseTransform(2.0, 3.0, logarithmic = true).model(fitted)
        val queries = DenseMatrix.ofRows(arrayOf(doubleArrayOf(0.0)))
        assertEquals(2.0, assertIs<LogResponseSurrogate>(wrapped).predictLog(queries).means[0])
        assertEquals(9.0, wrapped.predictLog(queries).variances[0])
        val conditioned = assertIs<ConditionableSurrogate>(wrapped).condition(
            TrainingData(
                queries,
                DenseVector.of(doubleArrayOf(exp(4.0))),
                DenseVector.of(doubleArrayOf(1.0)),
                DenseVector.of(doubleArrayOf(exp(8.0) * 9.0)),
            ),
        )
        assertEquals(2.0 / 3.0, response, 1e-12)
        assertEquals(1.0, variance, 1e-12)
        assertIs<LogResponseSurrogate>(conditioned)
    }

    @Test
    fun `log fitting with standardization predicts endpoint observations in original units`() {
        val space = Rate().compileSpace()
        val optimizer = RandomForestOptimizer(
            SearchTransform(space),
            config = ForestOptimizerConfig(forest = BatchForestConfig(nbrTrees = 1, bootstrap = false)),
            outcome = OutcomeSpec(responseScale = ResponseScale.Log, standardize = true),
        )
        val first = optimizer.suggest(2)
        optimizer.complete(first[0].id, 2.0)
        optimizer.complete(first[1].id, 8.0)
        assertEquals(2.0, optimizer.predict(first[0].sample), 1e-10)
        assertEquals(8.0, optimizer.predict(first[1].sample), 1e-10)
    }

    @Test
    fun `nonpositive log responses fail before recording or closing the trial`() {
        val optimizer = RandomForestOptimizer(
            SearchTransform(Rate().compileSpace()),
            outcome = OutcomeSpec(responseScale = ResponseScale.Log),
        )
        val trial = optimizer.suggest(1)[0]
        assertFailsWith<IllegalArgumentException> { optimizer.complete(trial.id, 0.0) }
        assertFailsWith<IllegalArgumentException> { optimizer.train(trial.sample, -1.0) }
        assertEquals(0L, optimizer.store.revision)
        assertEquals(TrialStatus.Pending, optimizer.trials.single().status)
    }
}
