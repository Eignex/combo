package com.eignex.combo.decisions

import com.eignex.combo.shared.Learner

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

    operator fun get(handle: FloatContextHandle): Double =
        floats[handle] ?: error("missing context value for float handle '${handle.name}'")

    fun isPresent(handle: FloatContextHandle): Boolean = handle in floats

    fun isPresent(handle: BoolContextHandle): Boolean = handle in bools
    fun isPresent(handle: IntContextHandle): Boolean = handle in ints

    companion object {
        val Empty: Context = Context(emptyMap(), emptyMap())
    }
}

class ContextBuilder internal constructor() {
    private val bools = mutableMapOf<BoolContextHandle, Boolean>()
    private val ints = mutableMapOf<IntContextHandle, Int>()
    private val floats = mutableMapOf<FloatContextHandle, Double>()

    fun set(handle: BoolContextHandle, value: Boolean) {
        bools[handle] = value
    }
    fun set(handle: IntContextHandle, value: Int) {
        ints[handle] = value
    }

    fun set(handle: FloatContextHandle, value: Double) {
        require(value.isFinite() && value in handle.min..handle.max) { "context '${handle.name}' outside its bounds" }
        floats[handle] = value
    }

    internal fun build() = Context(bools.toMap(), ints.toMap(), floats.toMap())
}

fun context(build: ContextBuilder.() -> Unit): Context = ContextBuilder().apply(build).build()
