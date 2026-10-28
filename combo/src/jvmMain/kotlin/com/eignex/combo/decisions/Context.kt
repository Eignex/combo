package com.eignex.combo.decisions

import com.eignex.combo.shared.Learner
import com.eignex.klause.arithmetic.polynomial.AlgebraicReal
import com.eignex.klause.arithmetic.real.ExactRealValue
import com.eignex.klause.arithmetic.real.toExactRealValue
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.util.BigInt
import com.eignex.klause.util.fitsLong
import com.eignex.klause.util.toLongExact

/**
 * Bag of contextual values supplied to a [com.eignex.combo.shared.Learner] at each `choose`/`update`
 * call. Built with [context].
 *
 * Every handle declared on a [DecisionSpace] must be present in the [Context] passed
 * to a [com.eignex.combo.decisions.FeatureEncoder.encode]; missing values raise.
 */
class Context internal constructor(
    internal val bools: Map<BoolContextHandle, Boolean>,
    internal val ints: Map<IntContextHandle, Int>,
    internal val floats: Map<FloatContextHandle, Double> = emptyMap(),
    internal val nativeAssumptions: com.eignex.klause.propagation.Assumptions =
        com.eignex.klause.propagation.Assumptions.None,
    internal val nativeReals: Map<Int, Double> = emptyMap(),
    internal val exactInts: Map<Int, BigInt> = emptyMap(),
    internal val exactReals: Map<Int, BigFraction> = emptyMap(),
    internal val exactRealValues: Map<Int, ExactRealValue> = emptyMap(),
) {
    operator fun get(handle: BoolContextHandle): Boolean = bools[handle] ?: if (handle.isOptional) {
        false
    } else {
        error("missing context value for bool handle '${handle.name}'")
    }

    operator fun get(handle: IntContextHandle): Int = ints[handle] ?: if (handle.isOptional) {
        handle.min
    } else {
        error("missing context value for int handle '${handle.name}'")
    }

    /** Read a required floating-point context in its original units. */
    operator fun get(handle: FloatContextHandle): Double =
        floats[handle] ?: error("missing context value for float handle '${handle.name}'")

    /** Whether a required floating-point context was supplied. */
    fun isPresent(handle: FloatContextHandle): Boolean = handle in floats

    fun isPresent(handle: BoolContextHandle): Boolean = handle in bools
    fun isPresent(handle: IntContextHandle): Boolean = handle in ints

    internal val hasRealValues: Boolean get() =
        floats.isNotEmpty() || nativeReals.isNotEmpty() || exactInts.isNotEmpty() || exactReals.isNotEmpty() ||
            exactRealValues.isNotEmpty()

    companion object {
        val Empty: Context = Context(emptyMap(), emptyMap())
    }
}

class ContextBuilder internal constructor() {
    private val bools = mutableMapOf<BoolContextHandle, Boolean>()
    private val ints = mutableMapOf<IntContextHandle, Int>()
    private val floats = mutableMapOf<FloatContextHandle, Double>()
    private var native = com.eignex.klause.propagation.Assumptions.None
    private val nativeReals = mutableMapOf<Int, Double>()
    private val exactInts = mutableMapOf<Int, BigInt>()
    private val exactReals = mutableMapOf<Int, BigFraction>()
    private val exactRealValues = mutableMapOf<Int, ExactRealValue>()

    fun set(handle: BoolContextHandle, value: Boolean) {
        bools[handle] = value
    }
    fun set(handle: IntContextHandle, value: Int) {
        ints[handle] = value
    }

    /** Set a finite floating-point context within its native bounds. */
    fun set(handle: FloatContextHandle, value: Double) {
        require(value.isFinite() && value in handle.min..handle.max) { "context '${handle.name}' outside its bounds" }
        floats[handle] = value
    }

    /** Pin a native Boolean column without requiring a schema handle. */
    fun set(id: com.eignex.klause.ir.SourceBoolId, value: Boolean) {
        native = native.withBool(id.index, value)
    }

    /** Pin a native integer column in its original Long units. */
    fun set(id: com.eignex.klause.ir.SourceIntId, value: Long) {
        native = native.withInt(id.index, value)
        exactInts.remove(id.index)
    }

    /** Pin an arbitrary-width integer using Klause's exact native arithmetic. */
    fun set(id: com.eignex.klause.ir.SourceIntId, value: BigInt) {
        if (value.fitsLong()) set(id, value.toLongExact()) else {
            native = com.eignex.klause.propagation.Assumptions(native.bools, native.ints - id.index)
            exactInts[id.index] = value
        }
    }

    /** Pin an exact rational real without rounding it through a Double. */
    fun set(id: com.eignex.klause.ir.SourceRealId, value: BigFraction) {
        exactReals[id.index] = value
        nativeReals.remove(id.index)
        exactRealValues.remove(id.index)
    }

    /** Pin an exact algebraic real using its native root identity. */
    fun set(id: com.eignex.klause.ir.SourceRealId, value: AlgebraicReal) = set(id, value.toExactRealValue())

    /** Pin an authoritative exact real expression without using its feature approximation. */
    fun set(id: com.eignex.klause.ir.SourceRealId, value: ExactRealValue) {
        exactRealValues[id.index] = value
        nativeReals.remove(id.index)
        exactReals.remove(id.index)
    }

    /** Pin a native real column without inventing finite bounds or a grid. */
    fun set(id: com.eignex.klause.ir.SourceRealId, value: Double) {
        require(value.isFinite()) { "real context must be finite" }
        nativeReals[id.index] = value
        exactReals.remove(id.index)
        exactRealValues.remove(id.index)
    }

    internal fun build() = Context(
        bools.toMap(), ints.toMap(), floats.toMap(), native, nativeReals.toMap(), exactInts.toMap(), exactReals.toMap(),
        exactRealValues.toMap(),
    )
}

fun context(build: ContextBuilder.() -> Unit): Context = ContextBuilder().apply(build).build()
