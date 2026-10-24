package com.eignex.combo.shared

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context
import com.eignex.kumulant.core.SeriesStat
import kotlin.math.abs

internal class LearningFeedback<Input>(
    private val encode: (BanditSample, Context) -> Input,
    private val mean: (Input) -> Double,
    private val record: (Input, Double, Double) -> Unit,
    private val rewards: SeriesStat<*>?,
    private val trainAbsError: SeriesStat<*>?,
    private val testAbsError: SeriesStat<*>?,
) {
    fun predict(sample: BanditSample, context: Context): Double = mean(encode(sample, context))

    fun train(sample: BanditSample, context: Context, reward: Double, weight: Double) =
        record(encode(sample, context), reward, weight)

    fun update(sample: BanditSample, context: Context, reward: Double, weight: Double, onRecorded: () -> Unit = {}) {
        updateEncoded(encode(sample, context), reward, weight, onRecorded)
    }

    fun updateEncoded(input: Input, reward: Double, weight: Double, onRecorded: () -> Unit = {}) {
        rewards?.update(reward, 0L, weight)
        testAbsError?.update(abs(reward - mean(input)), 0L, weight)
        record(input, reward, weight)
        onRecorded()
        trainAbsError?.update(abs(reward - mean(input)), 0L, weight)
    }
}
