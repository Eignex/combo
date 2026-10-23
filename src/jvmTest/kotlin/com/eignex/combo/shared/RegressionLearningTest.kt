package com.eignex.combo.shared

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context
import com.eignex.klause.solver.Sample
import com.eignex.koblas.DenseVector
import com.eignex.kumulant.stat.regression.glm.DiagonalRegressionStat
import com.eignex.kumulant.stat.summary.SumStat
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class RegressionLearningTest {
    @Test
    fun `update measures weighted errors before and after training using the same features`() {
        val regression = DiagonalRegressionStat(1)
        val rewards = SumStat()
        val before = SumStat()
        val after = SumStat()
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
        assertEquals(2.0 * abs(3.0 - initial), before.read().sum, 1e-12)
        assertEquals(2.0 * abs(3.0 - regression.read().predict(features)), after.read().sum, 1e-12)
    }

    @Test
    fun `off policy training leaves reward and error tracking untouched`() {
        val regression = DiagonalRegressionStat(1)
        val metrics = SumStat()
        val learning = RegressionLearning(
            regression,
            { _, _ -> DenseVector.of(doubleArrayOf(1.0)) },
            { regression.read().predict(it) },
            metrics,
            metrics,
            metrics,
        )
        val sample = BanditSample(Sample(booleanArrayOf(), longArrayOf()))
        learning.train(sample, Context.Empty, reward = 3.0, weight = 2.0)
        assertEquals(0.0, metrics.read().sum)
        assertTrue(learning.predict(sample) > 0.0)
    }
}
