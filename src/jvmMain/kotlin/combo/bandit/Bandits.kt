package combo.bandit

import com.eignex.klause.compile.CompiledProblem
import com.eignex.klause.solver.Sample
import com.eignex.klause.solver.Solver
import com.eignex.klause.solver.SolverParams
import com.eignex.kumulant.core.SeriesStat
import combo.decisions.BanditSample
import combo.decisions.CompiledDecisionSpace
import combo.util.RandomSequence
import kotlinx.serialization.Serializable

/** Thrown when a bandit cannot produce a feasible sample within its budget. */
class NoFeasibleSampleException(message: String) : RuntimeException(message)

/**
 * Per-bandit state that survives import/export. Concrete subtypes are `@Serializable`
 * data classes registered with a kotlinx-serialization `SerializersModule` when
 * polymorphic round-trips are needed; the interface itself is intentionally open so
 * downstream bandits can ship their own data without cross-package sealed-hierarchy
 * pain.
 *
 * Subtypes typically wrap kumulant `Result` snapshots: a [BanditData] is the union of
 * "what stats does this bandit hold" plus "which klause slot each one keys on", and
 * [remap] rewrites the latter when the schema changes.
 */
interface BanditData {
    /** Rewrite this data against a new schema. Stats whose slot is dropped should be discarded. */
    fun remap(slots: SlotRemap): BanditData
}

/**
 * Klause-aware slot remapping. The two namespaces (`bools` and `ints`) are independent;
 * a remapped int variable does not collide with a remapped bool variable of the same id.
 * Returning `-1` for an old slot means "drop this state".
 */
interface SlotRemap {
    fun remapBool(oldBoolId: Int): Int
    fun remapInt(oldIntId: Int): Int
    /** The destination [CompiledProblem]; consult its bookkeeping for size-derived rebuilds. */
    val newProblem: CompiledProblem
}

/**
 * A bandit optimises an online decision problem over a combinatorial domain. Each round
 * the user calls [choose] to get a feasible [BanditSample], plays it externally, then
 * reports the observed reward through [update]. The bandit owns the search strategy;
 * klause owns feasibility (samples come from a klause [Sampler] or
 * [com.eignex.klause.solver.Optimizer]).
 *
 * Float variables are bucketed in klause but the bandit emits continuous values: every
 * [BanditSample] returned by `choose`/`optimal` carries per-float dither drawn uniformly
 * within the chosen bucket. Models trained via [update] see those dithered values and
 * thus learn the continuous reward signal directly.
 *
 * Reward bookkeeping is delegated to an optional kumulant [SeriesStat] sink — if [rewards]
 * is non-null, every observed reward is folded in.
 */
interface Bandit<D : BanditData> {
    fun chooseOrThrow(): BanditSample
    fun choose(): BanditSample? = try { chooseOrThrow() } catch (_: NoFeasibleSampleException) { null }

    fun optimalOrThrow(): BanditSample
    fun optimal(): BanditSample? = try { optimalOrThrow() } catch (_: NoFeasibleSampleException) { null }

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
 * A [Bandit] backed by a learned predictive model. Adds [predict] (no side-effects) and
 * [train] (off-policy update without bumping reward bookkeeping).
 */
interface PredictionBandit<D : BanditData> : Bandit<D> {
    val trainAbsError: SeriesStat<*>?
    val testAbsError: SeriesStat<*>?

    fun predict(sample: BanditSample): Double

    fun train(sample: BanditSample, reward: Double, weight: Double = 1.0)
    fun trainAll(samples: List<BanditSample>, rewards: DoubleArray, weights: DoubleArray? = null) {
        require(samples.size == rewards.size) { "samples and rewards must have equal size" }
        require(weights == null || weights.size == rewards.size) { "weights must match rewards size" }
        for (i in samples.indices) train(samples[i], rewards[i], weights?.get(i) ?: 1.0)
    }
}

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
) : Bandit<RandomBanditData> {
    override val maximize: Boolean get() = true

    private val randomSequence = RandomSequence(randomSeed)

    override fun chooseOrThrow(): BanditSample {
        val s = sampler.sample(params)
            ?: throw NoFeasibleSampleException("klause sampler returned no feasible assignment")
        return BanditSample.dithered(s, space, randomSequence.next())
    }

    override fun optimalOrThrow(): BanditSample = chooseOrThrow()

    @Suppress("UNCHECKED_CAST")
    override fun update(sample: BanditSample, reward: Double, weight: Double) {
        (rewards as? SeriesStat<Any>)?.update(reward, 0L, weight)
    }

    override fun importData(data: RandomBanditData) {}
    override fun exportData(): RandomBanditData = RandomBanditData
}

@Serializable
data object RandomBanditData : BanditData {
    override fun remap(slots: SlotRemap): BanditData = this
}
