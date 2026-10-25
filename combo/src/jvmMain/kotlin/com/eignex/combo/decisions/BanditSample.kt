package com.eignex.combo.decisions

import com.eignex.klause.schema.FloatHandle
import com.eignex.klause.solver.Sample

/** Bandit-facing action preserving Klause's assignment and variable semantics. */
class BanditSample(
    /** The solver's unmodified assignment. */
    val sample: Sample,
) {
    /** Independent copy of the solver's Boolean coordinates. */
    val bools: BooleanArray get() = sample.bools

    /** Independent copy of the solver's integer coordinates. */
    val ints: LongArray get() = sample.ints

    /** Decode a float through the compiled Klause schema. */
    fun float(handle: FloatHandle, space: CompiledDecisionSpace): Double = space.compiled.decode(handle, sample)

    override fun equals(other: Any?): Boolean = other is BanditSample && sample == other.sample
    override fun hashCode(): Int = sample.hashCode()
}
