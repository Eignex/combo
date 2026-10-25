package com.eignex.combo.shared

import com.eignex.combo.decisions.BanditSample
import com.eignex.klause.compile.CompiledSchema
import com.eignex.klause.solver.Solver
import com.eignex.kumulant.core.PairedStat
import com.eignex.kumulant.core.SeriesStat

/** Thrown when a bandit cannot produce a feasible sample within its budget. */
class NoFeasibleSampleException(message: String) : RuntimeException(message)

/**
 * Per-learner state that survives import/export. Concrete subtypes can be `@Serializable`
 * data classes registered with a kotlinx-serialization `SerializersModule` when
 * polymorphic round-trips are needed; the interface itself is intentionally open so
 * downstream learners can ship their own data without cross-package sealed-hierarchy
 * pain.
 *
 * Subtypes typically wrap kumulant `Result` snapshots: a [LearnerData] is the union
 * of "what stats does this learner hold" plus "which klause slot each one keys on",
 * and [remap] rewrites the latter when the schema changes.
 */
interface LearnerData {
    /** Rewrite this data against a new schema. Stats whose slot is dropped should be discarded. */
    fun remap(slots: SlotRemap): LearnerData
}

/**
 * Klause-aware slot remapping. The two namespaces (`bools` and `ints`) are independent;
 * a remapped int variable does not collide with a remapped bool variable of the same id.
 * Returning `-1` for an old slot means "drop this state".
 */
interface SlotRemap {
    fun remapBool(oldBoolId: Int): Int
    fun remapInt(oldIntId: Int): Int

    /** The destination [CompiledSchema]; consult its bookkeeping for size-derived rebuilds. */
    val newProblem: CompiledSchema
}

/**
 * Configuration learner over a constrained decision space. Streaming bandits update
 * sufficient statistics and incremental models; history-based optimizers retain
 * observations and refit from them. Both expose the same choose / play / update loop.
 *
 * Each round the caller invokes [choose] to get a feasible [BanditSample], plays it
 * externally, then reports the observed reward through [update]. The learner owns the
 * search strategy; klause owns feasibility (samples come from a klause [Solver] or
 * [com.eignex.klause.solver.Optimizer]).
 *
 * Variables keep Klause's types, constraints, and decoded values. Context variables
 * are ordinary Klause variables pinned to caller-supplied values for each decision.
 *
 * Reward bookkeeping is delegated to an optional kumulant [SeriesStat] sink: if
 * [rewards] is non-null, every observed reward is folded in.
 */
interface Learner<D : LearnerData> {
    fun chooseOrThrow(): BanditSample
    fun choose(): BanditSample? = try {
        chooseOrThrow()
    } catch (_: NoFeasibleSampleException) {
        null
    }

    fun optimalOrThrow(): BanditSample
    fun optimal(): BanditSample? = try {
        optimalOrThrow()
    } catch (_: NoFeasibleSampleException) {
        null
    }

    fun update(sample: BanditSample, reward: Double, weight: Double = 1.0)
    fun updateAll(samples: List<BanditSample>, rewards: DoubleArray, weights: DoubleArray? = null) {
        require(samples.size == rewards.size) { "samples and rewards must have equal size" }
        require(weights == null || weights.size == rewards.size) { "weights must match rewards size" }
        for (i in samples.indices) update(samples[i], rewards[i], weights?.get(i) ?: 1.0)
    }

    fun importData(data: D)
    fun exportData(): D

    val randomSeed: Int
    val maximize: Boolean

    /** Optional kumulant sink that aggregates every observed reward. */
    val rewards: SeriesStat<*>?
}

/**
 * A [Learner] backed by a learned predictive model. Adds [predict] (no side-effects)
 * and [train] (off-policy update without bumping reward bookkeeping).
 */
interface PredictionLearner<D : LearnerData> : Learner<D> {
    /** Optional scorer receiving (prediction, reward) after learning, with the observation weight. */
    val trainScore: PairedStat<*>?

    /** Optional scorer receiving (prediction, reward) before learning, with the observation weight. */
    val testScore: PairedStat<*>?

    fun predict(sample: BanditSample): Double

    fun train(sample: BanditSample, reward: Double, weight: Double = 1.0)
    fun trainAll(samples: List<BanditSample>, rewards: DoubleArray, weights: DoubleArray? = null) {
        require(samples.size == rewards.size) { "samples and rewards must have equal size" }
        require(weights == null || weights.size == rewards.size) { "weights must match rewards size" }
        for (i in samples.indices) train(samples[i], rewards[i], weights?.get(i) ?: 1.0)
    }
}
