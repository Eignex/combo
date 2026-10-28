package com.eignex.combo.bandit.dt

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.FloatHandle
import com.eignex.klause.schema.IntHandle
import com.eignex.klause.schema.NominalHandle

/**
 * Tree bandit's view of a klause [com.eignex.klause.solver.Sample]. Provides typed access by handle (no
 * bit-blasted indexing) and presence checks for optional variables.
 *
 * Trees ask the row simple questions during split evaluation:
 *  - "what's `model.budget`'s value here?" via [int]
 *  - "is `model.audio` active in this row?" via [isPresent]
 *
 * Implementations are expected to be cheap, ideally just a wrapper around the
 * underlying [com.eignex.klause.solver.Sample] — split metrics walk many rows in tight loops.
 */
interface TreeRow {
    val space: CompiledDecisionSpace

    fun bool(handle: BoolHandle): Boolean
    fun int(handle: IntHandle): Long
    fun nominal(handle: NominalHandle): String

    /** Float value decoded by Klause in its declared units. */
    fun float(handle: FloatHandle): Double

    fun isPresent(handle: BoolHandle): Boolean
    fun isPresent(handle: IntHandle): Boolean
    fun isPresent(handle: NominalHandle): Boolean
    fun isPresent(handle: FloatHandle): Boolean

    val sample: BanditSample
}

internal class SampleTreeRow(override val space: CompiledDecisionSpace, override val sample: BanditSample) : TreeRow {
    override fun bool(handle: BoolHandle): Boolean = space.coordinates.decode(handle, sample.sample)
    override fun int(handle: IntHandle): Long = space.coordinates.decode(handle, sample.sample)
    override fun nominal(handle: NominalHandle): String = space.coordinates.decode(handle, sample.sample)

    override fun float(handle: FloatHandle): Double = sample.float(handle, space)

    override fun isPresent(handle: BoolHandle): Boolean = space.isActive(handle, sample.sample)
    override fun isPresent(handle: IntHandle): Boolean = space.isActive(handle, sample.sample)
    override fun isPresent(handle: NominalHandle): Boolean = space.isActive(handle, sample.sample)
    override fun isPresent(handle: FloatHandle): Boolean {
        // Presence is keyed by name across Klause variable representations.
        val cond = space.activeConditions[handle.name] ?: return true
        return space.evaluate(cond, sample.sample)
    }
}
