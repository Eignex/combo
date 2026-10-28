package com.eignex.combo.bandit

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.shared.Learner
import com.eignex.combo.shared.LearnerData
import com.eignex.combo.shared.NoFeasibleSampleException
import com.eignex.combo.shared.SlotRemap
import com.eignex.klause.solver.Solver
import com.eignex.klause.solver.SolverParams
import com.eignex.klause.solver.result.SampleResult
import com.eignex.kumulant.core.SeriesStat
import kotlinx.serialization.Serializable

/**
 * Baseline bandit that delegates sampling to a klause [Solver] and ignores rewards.
 * Useful as an A/B control or for warm-starting more expensive bandits.
 */
class RandomBandit<P : SolverParams>(
    val space: CompiledDecisionSpace,
    val sampler: Solver<P>,
    val params: P,
    override val randomSeed: Int = 0,
    override val rewards: SeriesStat<*>? = null,
) : Learner<RandomLearnerData> {
    override val maximize: Boolean get() = true

    override fun chooseOrThrow(): BanditSample {
        val s = (sampler.sample(params) as? SampleResult.Found)?.sample
            ?: throw NoFeasibleSampleException("klause sampler returned no feasible assignment")
        return BanditSample(s)
    }

    override fun optimalOrThrow(): BanditSample = chooseOrThrow()

    override fun update(sample: BanditSample, reward: Double, weight: Double) {
        rewards?.update(reward, 0L, weight)
    }

    override fun importData(data: RandomLearnerData) {}
    override fun exportData(): RandomLearnerData = RandomLearnerData
}

@Serializable
data object RandomLearnerData : LearnerData {
    override fun remap(slots: SlotRemap): LearnerData = this
}
