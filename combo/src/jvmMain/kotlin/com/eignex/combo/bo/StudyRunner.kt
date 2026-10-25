package com.eignex.combo.bo

import com.eignex.combo.decisions.Context
import com.eignex.combo.shared.WitnessRequest
import com.eignex.combo.shared.WitnessSearch
import kotlin.random.Random

/** Model-independent lifecycle adapter, also useful for baseline suggestion policies. */
class StudyRunner(
    /** Lifecycle and recorded observations. */
    val study: Study,
    /** Pure batch proposal algorithm. */
    val policy: SuggestionPolicy,
) : TrialOptimizer {
    override val trials: List<Trial> get() = study.trials
    override fun suggest(count: Int, context: Context): TrialBatch = study.suggest(policy, count, context)
    override fun complete(id: Long, reward: Double, weight: Double, noiseVariance: Double?) =
        study.complete(id, reward, weight, noiseVariance)
    override fun fail(id: Long, reason: String) = study.fail(id, reason)
    override fun cancel(id: Long, reason: String?) = study.cancel(id, reason)
}

/** Feasible random-search baseline without surrogate fitting or acquisition optimization. */
class RandomSuggestionPolicy(
    /** Original decision space, including hard contextual variables. */
    val space: com.eignex.combo.decisions.CompiledDecisionSpace,
    /** Feasible random initialization provider. */
    val witnesses: WitnessSearch,
    /** Seed mixed with the persisted study round number. */
    val seed: Int = 0,
) : SuggestionPolicy {
    override fun suggest(snapshot: StudySnapshot, count: Int, context: Context): ProposalBatch {
        val batch = witnesses.witnesses(
            WitnessRequest(
                count,
                space.assumptionsFor(context),
                exclude = snapshot.pending.map { it.sample.sample },
            ),
            Random(seed.toLong() xor snapshot.round),
        )
        return ProposalBatch(batch.witnesses.map { com.eignex.combo.decisions.BanditSample(it) }, batch.termination)
    }
}
