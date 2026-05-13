package combo.bandit.dt

import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.IntHandle
import com.eignex.klause.schema.NominalHandle
import com.eignex.klause.solver.Sample
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

    fun isPresent(handle: BoolHandle): Boolean
    fun isPresent(handle: IntHandle): Boolean
    fun isPresent(handle: NominalHandle): Boolean

    val sample: Sample
}

internal class SampleTreeRow(
    override val space: CompiledDecisionSpace,
    override val sample: Sample,
) : TreeRow {
    override fun bool(handle: BoolHandle): Boolean = space.compiled.decode(handle, sample)
    override fun int(handle: IntHandle): Int = space.compiled.decode(handle, sample)
    override fun nominal(handle: NominalHandle): String = space.compiled.decode(handle, sample)

    override fun isPresent(handle: BoolHandle): Boolean = space.isActive(handle, sample)
    override fun isPresent(handle: IntHandle): Boolean = space.isActive(handle, sample)
    override fun isPresent(handle: NominalHandle): Boolean = space.isActive(handle, sample)
}
