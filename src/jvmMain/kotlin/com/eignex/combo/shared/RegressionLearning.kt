package com.eignex.combo.shared

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context
import com.eignex.koblas.Vector
import com.eignex.kumulant.core.RegressionStat
import com.eignex.kumulant.core.Result
import com.eignex.kumulant.core.SeriesStat

internal class RegressionLearning<R : Result>(
    private val regression: RegressionStat<R>,
    encode: (BanditSample, Context) -> Vector,
    mean: (Vector) -> Double,
    rewards: SeriesStat<*>?,
    trainAbsError: SeriesStat<*>?,
    testAbsError: SeriesStat<*>?,
) {
    private val feedback = LearningFeedback(encode, mean, regression::update, rewards, trainAbsError, testAbsError)

    fun predict(sample: BanditSample, context: Context = Context.Empty): Double = feedback.predict(sample, context)

    fun train(sample: BanditSample, context: Context, reward: Double, weight: Double) =
        feedback.train(sample, context, reward, weight)

    fun update(sample: BanditSample, context: Context, reward: Double, weight: Double) =
        feedback.update(sample, context, reward, weight)

    fun snapshot(): R = regression.read()
    fun merge(state: R) = regression.merge(state)
}
