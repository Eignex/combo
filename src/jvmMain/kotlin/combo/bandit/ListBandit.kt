package combo.bandit

import com.eignex.klause.solver.Sample
import com.eignex.kumulant.core.Result
import com.eignex.kumulant.core.SeriesStat
import combo.bandit.univariate.BanditPolicy
import combo.decisions.CompiledDecisionSpace
import combo.decisions.Context
import combo.util.RandomSequence
import java.util.concurrent.atomic.AtomicLong
import kotlinx.serialization.Serializable

/**
 * Bandit over a pre-defined list of klause [Sample]s. Each sample is an arm with its
 * own [BanditPolicy]-owned accumulator; [chooseOrThrow] picks the best-scoring arm
 * among those compatible with the round's [Context] (i.e. whose pinned-context bits
 * match the caller-supplied assumptions).
 *
 * Use a klause [com.eignex.klause.solver.Sampler.enumerate] to populate [samples] from
 * a [combo.decisions.DecisionSpace] when you don't want to hand-curate them.
 *
 * @param samples the arms; should be feasible against the [space]'s constraints.
 * @param policy decides which arm to play and owns each arm's [SeriesStat].
 * @param space resolves [Context] to klause [com.eignex.klause.solver.Assumptions]
 *              and matches them against each sample.
 */
class ListBandit<R : Result>(
    val samples: List<Sample>,
    val policy: BanditPolicy<R>,
    val space: CompiledDecisionSpace,
    override val randomSeed: Int = System.currentTimeMillis().toInt(),
    override val maximize: Boolean = true,
    override val rewards: SeriesStat<*>? = null,
) : Bandit<ListBanditData> {

    init {
        require(samples.isNotEmpty()) { "ListBandit needs at least one sample arm" }
    }

    private val arms: Array<SeriesStat<R>> = Array(samples.size) {
        policy.createArm().also { stat -> policy.addArm(stat.read(0L)) }
    }
    private val randomSequence = RandomSequence(randomSeed)
    private val step = AtomicLong()

    fun chooseOrThrow(context: Context): Sample {
        val assumptions = space.assumptionsFor(context)
        val rng = randomSequence.next()
        val t = step.getAndIncrement()
        var bestIdx = -1
        var bestScore = Double.NEGATIVE_INFINITY
        for (i in samples.indices) {
            if (!space.matches(samples[i], assumptions)) continue
            val score = policy.evaluate(arms[i].read(0L), t, maximize, rng)
            if (score > bestScore) { bestScore = score; bestIdx = i }
        }
        if (bestIdx < 0) throw NoFeasibleSampleException(
            "no sample in the list matches the supplied context assumptions",
        )
        return samples[bestIdx]
    }

    fun choose(context: Context): Sample? = try {
        chooseOrThrow(context)
    } catch (_: NoFeasibleSampleException) {
        null
    }

    fun optimalOrThrow(context: Context): Sample {
        // "Optimal" = exploit best-mean arm. We use a fixed-seed RNG so the answer is
        // deterministic across rounds; the policy's evaluate is called once per arm.
        val assumptions = space.assumptionsFor(context)
        val rng = kotlin.random.Random(0L)
        val t = step.get()
        var bestIdx = -1
        var bestScore = Double.NEGATIVE_INFINITY
        for (i in samples.indices) {
            if (!space.matches(samples[i], assumptions)) continue
            val score = policy.evaluate(arms[i].read(0L), t, maximize, rng)
            if (score > bestScore) { bestScore = score; bestIdx = i }
        }
        if (bestIdx < 0) throw NoFeasibleSampleException(
            "no sample in the list matches the supplied context assumptions",
        )
        return samples[bestIdx]
    }

    fun update(sample: Sample, context: Context, reward: Double, weight: Double = 1.0) {
        val idx = samples.indexOf(sample)
        if (idx >= 0) policy.update(arms[idx], reward, weight)
        @Suppress("UNCHECKED_CAST")
        (rewards as? SeriesStat<Any>)?.update(reward, 0L, weight)
    }

    // Bandit<D> interface — contextless overloads default to empty context.
    override fun chooseOrThrow(): Sample = chooseOrThrow(Context.Empty)
    override fun optimalOrThrow(): Sample = optimalOrThrow(Context.Empty)
    override fun update(sample: Sample, reward: Double, weight: Double) =
        update(sample, Context.Empty, reward, weight)

    /** Snapshot the current per-arm stats. */
    fun snapshot(): List<R> = arms.map { it.read(0L) }

    override fun importData(data: ListBanditData) { /* no-op for now */ }
    override fun exportData(): ListBanditData = ListBanditData
}

@Serializable
data object ListBanditData : BanditData {
    override fun remap(slots: SlotRemap): BanditData = this
}
