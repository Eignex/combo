package combo.decisions

import com.eignex.klause.schema.FloatHandle
import com.eignex.klause.solver.Sample
import kotlin.random.Random

/**
 * Bandit-facing view of an action. Wraps klause's bucket-quantized [Sample] with
 * per-float **continuous** values sampled uniformly within each bucket.
 *
 * Why dither: klause solves over discrete bucket grids — necessary for constraint
 * checking, sufficient for feasibility — but the bandit operates in real-value
 * Double space. Returning a bucket-midpoint as the action would systematically
 * bias training data toward those midpoints; dithering uniformly within the
 * bucket sidesteps the bias and gives the bandit sub-bucket output resolution
 * for free. Klause constraints are bucket-precision, so any value inside the
 * bucket is still feasible by construction.
 *
 * Reads:
 *  - [bools] / [ints] surface the underlying klause sample unchanged.
 *  - [float] returns the dithered Double; falls back to klause's bucket-midpoint
 *    decode when this `BanditSample` was constructed without dither.
 *
 * `BanditSample.equals/hashCode` ignore the dither array intentionally — equality is
 * a bucket-level concept, used e.g. by tree routing (`tree.findLeaf(row) === leaf`).
 */
class BanditSample internal constructor(
    val sample: Sample,
    private val ditheredFloats: DoubleArray,
) {
    /** Underlying packed bools, identical to [sample.bools]. */
    val bools: BooleanArray get() = sample.bools

    /** Underlying packed ints, identical to [sample.ints] — bucket indices for float vars. */
    val ints: IntArray get() = sample.ints

    /** Decoded continuous float value for [handle]. Returns the dithered value when this
     *  sample carries one; otherwise falls back to klause's bucket-midpoint decode. */
    fun float(handle: FloatHandle, space: CompiledDecisionSpace): Double {
        val id = space.compiled.intVarIdByName[handle.name]
            ?: error("BanditSample.float: unknown float handle '${handle.name}'")
        if (id < ditheredFloats.size) {
            val v = ditheredFloats[id]
            if (!v.isNaN()) return v
        }
        return space.compiled.decode(handle, sample)
    }

    override fun equals(other: Any?): Boolean = other is BanditSample && sample == other.sample
    override fun hashCode(): Int = sample.hashCode()

    companion object {
        /** Wrap [sample] without dither — every float reads back as its bucket midpoint.
         *  Useful for tests, deterministic playback, and external samples whose dither
         *  history is unknown. */
        fun undithered(sample: Sample): BanditSample = BanditSample(sample, EMPTY)

        /**
         * Wrap [sample] with fresh per-float dither drawn from [rng]. Each float's
         * value is sampled uniformly in `[center - bucketWidth/2, center + bucketWidth/2]`
         * clamped to the handle's `[min, max]` (handles boundary buckets cleanly).
         */
        fun dithered(sample: Sample, space: CompiledDecisionSpace, rng: Random): BanditSample {
            val n = space.compiled.problem.numIntVars
            if (space.compiled.floatDecoders.isEmpty()) return BanditSample(sample, EMPTY)
            val out = DoubleArray(n) { Double.NaN }
            for ((name, spec) in space.compiled.floatDecoders) {
                val id = space.compiled.intVarIdByName[name] ?: continue
                val bucket = sample.ints[id]
                val span = spec.max - spec.min
                val divisor = (spec.buckets - 1).coerceAtLeast(1).toDouble()
                val width = span / divisor
                val center = spec.min + bucket.toDouble() * width
                out[id] = (center + (rng.nextDouble() - 0.5) * width).coerceIn(spec.min, spec.max)
            }
            return BanditSample(sample, out)
        }

        private val EMPTY = DoubleArray(0)
    }
}
