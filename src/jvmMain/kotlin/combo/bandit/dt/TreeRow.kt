package combo.bandit.dt

import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.FloatHandle
import com.eignex.klause.schema.IntHandle
import com.eignex.klause.schema.NominalHandle
import combo.decisions.BanditSample
import combo.decisions.CompiledDecisionSpace

/**
 * Tree bandit's view of a klause [Sample]. Provides typed access by handle (no
 * bit-blasted indexing) and presence checks for optional variables.
 *
 * Trees ask the row simple questions during split evaluation:
 *  - "what's `model.budget`'s value here?" via [int]
 *  - "is `model.audio` active in this row?" via [isPresent]
 *
 * Implementations are expected to be cheap, ideally just a wrapper around the
 * underlying [Sample] — split metrics walk many rows in tight loops.
 */
interface TreeRow {
    val space: CompiledDecisionSpace

    fun bool(handle: BoolHandle): Boolean
    fun int(handle: IntHandle): Int
    fun nominal(handle: NominalHandle): String
    /** Decoded value of a float variable in **real Double units**. Klause stores
     *  floats as bucketed integers internally; this returns the reconstructed
     *  Double so tree splits and surrogate features speak the same language the
     *  user defined the variable in. */
    fun float(handle: FloatHandle): Double

    fun isPresent(handle: BoolHandle): Boolean
    fun isPresent(handle: IntHandle): Boolean
    fun isPresent(handle: NominalHandle): Boolean
    fun isPresent(handle: FloatHandle): Boolean

    val sample: BanditSample
}

internal class SampleTreeRow(
    override val space: CompiledDecisionSpace,
    override val sample: BanditSample,
) : TreeRow {
    override fun bool(handle: BoolHandle): Boolean = space.compiled.decode(handle, sample.sample)
    override fun int(handle: IntHandle): Int = space.compiled.decode(handle, sample.sample)
    override fun nominal(handle: NominalHandle): String = space.compiled.decode(handle, sample.sample)
    /** Reads the dithered continuous value when the sample carries one; falls back to
     *  klause's bucket-midpoint decode otherwise. The dither layer is transparent to
     *  the rest of the tree code — splits compare against real values either way. */
    override fun float(handle: FloatHandle): Double = sample.float(handle, space)

    override fun isPresent(handle: BoolHandle): Boolean = space.isActive(handle, sample.sample)
    override fun isPresent(handle: IntHandle): Boolean = space.isActive(handle, sample.sample)
    override fun isPresent(handle: NominalHandle): Boolean = space.isActive(handle, sample.sample)
    override fun isPresent(handle: FloatHandle): Boolean =
        // Float handles share klause's int-id space — the active condition is keyed on the same name.
        space.activeConditions[handle.name] == null || space.isActive(
            com.eignex.klause.schema.IntHandle(handle.name, 0, handle.buckets - 1), sample.sample,
        )
}
