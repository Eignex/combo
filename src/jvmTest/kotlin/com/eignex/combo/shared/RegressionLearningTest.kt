package com.eignex.combo.shared

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context
import com.eignex.klause.solver.Sample
import com.eignex.koblas.DenseVector
import com.eignex.kumulant.stat.regression.glm.DiagonalRegressionStat
import com.eignex.kumulant.stat.score.LogLossStat
import com.eignex.kumulant.stat.score.MaeLossStat
import com.eignex.kumulant.stat.score.MseLossStat
import com.eignex.kumulant.stat.summary.SumStat
import kotlin.math.abs
import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class RegressionLearningTest {
    @Test
    fun `update measures weighted errors before and after training using the same features`() {
        val regression = DiagonalRegressionStat(1)
        val rewards = SumStat()
        val before = MseLossStat()
        val after = MaeLossStat()
        var encodings = 0
        val features = DenseVector.of(doubleArrayOf(1.0))
        val learning = RegressionLearning(
            regression,
            { _, _ ->
                encodings++
                features
            },
            { regression.read().predict(it) },
            rewards,
            after,
            before,
        )
        val sample = BanditSample(Sample(booleanArrayOf(), longArrayOf()))
        val initial = regression.read().predict(features)
        learning.update(sample, Context.Empty, reward = 3.0, weight = 2.0)
        assertEquals(1, encodings)
        assertEquals(6.0, rewards.read().sum)
        assertEquals((3.0 - initial) * (3.0 - initial), before.read().mean, 1e-12)
        assertEquals(2.0, before.read().totalWeights)
        assertEquals(abs(3.0 - regression.read().predict(features)), after.read().mean, 1e-12)
        assertEquals(2.0, after.read().totalWeights)
    }

    @Test
    fun `off policy training leaves reward and error tracking untouched`() {
        val regression = DiagonalRegressionStat(1)
        val rewards = SumStat()
        val metrics = MaeLossStat()
        val learning = RegressionLearning(
            regression,
            { _, _ -> DenseVector.of(doubleArrayOf(1.0)) },
            { regression.read().predict(it) },
            rewards,
            metrics,
            metrics,
        )
        val sample = BanditSample(Sample(booleanArrayOf(), longArrayOf()))
        learning.train(sample, Context.Empty, reward = 3.0, weight = 2.0)
        assertEquals(0.0, rewards.read().sum)
        assertEquals(0.0, metrics.read().totalWeights)
        assertTrue(learning.predict(sample) > 0.0)
    }

    @Test
    fun `feedback passes predictions before outcomes to weighted paired scorers`() {
        var prediction = 0.2
        val before = LogLossStat()
        val after = MseLossStat()
        val feedback = LearningFeedback(
            encode = { _: BanditSample, _: Context -> Unit },
            mean = { _: Unit -> prediction },
            record = { _: Unit, _: Double, _: Double -> prediction = 0.8 },
            rewards = null,
            trainScore = after,
            testScore = before,
        )
        val sample = BanditSample(Sample(booleanArrayOf(), longArrayOf()))

        feedback.update(sample, Context.Empty, reward = 1.0, weight = 2.0)
        feedback.update(sample, Context.Empty, reward = 0.0, weight = 1.0)

        assertEquals(-ln(0.2), before.read().mean, 1e-12)
        assertEquals(0.24, after.read().mean, 1e-12)
        assertEquals(3.0, before.read().totalWeights)
        assertEquals(3.0, after.read().totalWeights)
    }
}
