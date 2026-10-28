@file:OptIn(kotlin.concurrent.atomics.ExperimentalAtomicApi::class)

package com.eignex.combo.bandit

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.Context
import com.eignex.combo.shared.Learner
import com.eignex.combo.shared.LearnerData
import com.eignex.combo.shared.NoFeasibleSampleException
import com.eignex.combo.shared.SlotRemap
import com.eignex.combo.util.RandomSequence
import com.eignex.klause.solver.Sample
import com.eignex.kumulant.bandit.univariate.BanditPolicy
import com.eignex.kumulant.core.Result
import com.eignex.kumulant.core.SeriesStat
import kotlinx.serialization.Serializable
import kotlin.concurrent.atomics.AtomicLong

/**
 * Bandit over a pre-defined list of klause [Sample]s. Each sample is an arm with its
 * own [BanditPolicy]-owned accumulator; [chooseOrThrow] picks the best-scoring arm
 * among those compatible with the round's [Context] (i.e. whose pinned-context bits
 * match the caller-supplied assumptions).
 *
 * Use a klause [com.eignex.klause.solver.Solver.enumerate] to populate [samples] from
 * a [com.eignex.combo.decisions.DecisionSpace] when you don't want to hand-curate them.
 *
 * @param samples the arms; should be feasible against the [space]'s constraints.
 * @param policy decides which arm to play and owns each arm's [SeriesStat].
 * @param space resolves [Context] to klause [com.eignex.klause.propagation.Assumptions]
 *              and matches them against each sample.
 */
class ListBandit<R : Result>(
    val samples: List<Sample>,
    val policy: BanditPolicy<R>,
    val space: CompiledDecisionSpace,
    override val randomSeed: Int = kotlin.time.Clock.System.now().toEpochMilliseconds().toInt(),
    override val maximize: Boolean = true,
    override val rewards: SeriesStat<*>? = null,
) : Learner<ListBanditData> {

    init {
        require(samples.isNotEmpty()) { "ListBandit needs at least one sample arm" }
    }

    private val arms: Array<SeriesStat<R>> = Array(samples.size) {
        policy.createArm().also { stat -> policy.addArm(stat.read(0L)) }
    }
    private val randomSequence = RandomSequence(randomSeed)
    private val step = AtomicLong(0L)

    /** Flip a policy score for minimization; kumulant's [policy] now scores
     *  "higher is better" and no longer takes a maximize flag. */
    private fun signed(m: Double): Double = if (maximize) m else -m

    fun chooseOrThrow(context: Context): BanditSample {
        val assumptions = space.assumptionsFor(context)
        val rng = randomSequence.next()
        val t = step.fetchAndAdd(1L)
        var bestIdx = -1
        var bestScore = Double.NEGATIVE_INFINITY
        for (i in samples.indices) {
            if (!space.matches(samples[i], assumptions)) continue
            val score = signed(policy.evaluate(arms[i].read(0L), t, rng))
            if (score > bestScore) {
                bestScore = score
                bestIdx = i
            }
        }
        if (bestIdx < 0) {
            throw NoFeasibleSampleException(
                "no sample in the list matches the supplied context assumptions",
            )
        }
        return BanditSample(samples[bestIdx])
    }

    fun choose(context: Context): BanditSample? = try {
        chooseOrThrow(context)
    } catch (_: NoFeasibleSampleException) {
        null
    }

    fun optimalOrThrow(context: Context): BanditSample {
        // "Optimal" = exploit best-mean arm. We use a fixed-seed RNG so the answer is
        // deterministic across rounds; the policy's evaluate is called once per arm.
        val assumptions = space.assumptionsFor(context)
        val rng = kotlin.random.Random(0L)
        val t = step.load()
        var bestIdx = -1
        var bestScore = Double.NEGATIVE_INFINITY
        for (i in samples.indices) {
            if (!space.matches(samples[i], assumptions)) continue
            val score = signed(policy.evaluate(arms[i].read(0L), t, rng))
            if (score > bestScore) {
                bestScore = score
                bestIdx = i
            }
        }
        if (bestIdx < 0) {
            throw NoFeasibleSampleException(
                "no sample in the list matches the supplied context assumptions",
            )
        }
        return BanditSample(samples[bestIdx])
    }

    fun update(sample: BanditSample, context: Context, reward: Double, weight: Double = 1.0) {
        // Arms use the identity of the underlying Klause sample.
        val idx = samples.indexOf(sample.sample)
        if (idx >= 0) policy.update(arms[idx], reward, weight)
        rewards?.update(reward, 0L, weight)
    }

    // Learner<D> interface — contextless overloads default to empty context.
    override fun chooseOrThrow(): BanditSample = chooseOrThrow(Context.Empty)
    override fun optimalOrThrow(): BanditSample = optimalOrThrow(Context.Empty)
    override fun update(sample: BanditSample, reward: Double, weight: Double) =
        update(sample, Context.Empty, reward, weight)

    /** Snapshot the current per-arm stats. */
    fun snapshot(): List<R> = arms.map { it.read(0L) }

    override fun importData(data: ListBanditData) { /* no-op for now */ }
    override fun exportData(): ListBanditData = ListBanditData
}

@Serializable
data object ListBanditData : LearnerData {
    override fun remap(slots: SlotRemap): LearnerData = this
}
