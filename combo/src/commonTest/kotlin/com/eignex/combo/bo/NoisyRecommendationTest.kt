package com.eignex.combo.bo

import com.eignex.combo.bo.model.GaussianProcessConfig
import com.eignex.combo.decisions.Context
import com.eignex.combo.shared.assignmentKey
import com.eignex.combo.test.TimedTest
import kotlin.test.Test
import kotlin.test.assertEquals

class NoisyRecommendationTest : TimedTest() {
    @Test
    fun `repeated recommendations preserve the study and next suggestion random stream`() {
        val problem = refinementProblem("noisy")
        fun optimizer() = BayesianOptimizer(
            SearchTransform(problem.space),
            GaussianProcessSurrogate(GaussianProcessConfig(hyperparameterTrials = 0)),
            randomSeed = 17,
            acquisitionOptimizer = LocalAndRandomSearch(
                SearchTransform(problem.space),
                noisyWitnesses(),
                candidateCount = 4,
                localSteps = 0,
            ),
            outcome = OutcomeSpec(maximize = false, noise = ObservationNoise.Noisy),
            initialDesign = InitialDesign { _, _ -> false },
        ).also { it.train(continuousPoint(doubleArrayOf(0.4, 0.6)), Context.Empty, 1.1, noiseVariance = 0.01) }
        val inspected = optimizer()
        val untouched = optimizer()
        val before = inspected.study.snapshot()

        val first = inspected.recommendOrThrow()
        val second = inspected.recommendOrThrow()
        inspected.recommendOrThrow(
            LocalAndRandomSearch(SearchTransform(problem.space), noisyWitnesses(), candidateCount = 8, localSteps = 0),
        )

        assertEquals(first.sample.assignmentKey(), second.sample.assignmentKey())
        assertEquals(before.round, inspected.study.snapshot().round)
        assertEquals(before.revision, inspected.study.snapshot().revision)
        assertEquals(before.history, inspected.study.snapshot().history)
        assertEquals(
            untouched.suggest(1)[0].sample.sample.assignmentKey(),
            inspected.suggest(1)[0].sample.sample.assignmentKey(),
        )
    }
}
