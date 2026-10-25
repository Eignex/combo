package com.eignex.combo.decisions

import com.eignex.klause.model.FloatSpec
import com.eignex.klause.model.IntSpec
import kotlinx.serialization.Serializable
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.ln1p

/** Geometry used for numeric proposals and model features; solver values retain their original units. */
@Serializable
enum class Scale {
    /** Equal attention to equal additive intervals. */
    Linear,

    /** Equal attention to equal ratios; requires a strictly positive lower bound. */
    Log,

    /** Increasing resolution near the upper bound using -log(1 + max - value). */
    ReverseLog,
}

internal data class NumericDomain(val min: Double, val max: Double, val scale: Scale) {
    init {
        require(min.isFinite() && max.isFinite() && min <= max) { "invalid numeric bounds" }
        require(scale != Scale.Log || min > 0.0) { "log scaling requires positive bounds" }
        require(scale != Scale.ReverseLog || (max - min).isFinite()) { "reverse-log range is too wide" }
    }

    fun feature(value: Double): Double = when (scale) {
        Scale.Linear -> value
        Scale.Log -> ln(value)
        Scale.ReverseLog -> -ln1p(max - value)
    }

    fun fromUnit(unit: Double): Double {
        require(unit in 0.0..1.0) { "unit coordinate must lie in [0, 1]" }
        if (unit == 0.0 || min == max) return min
        if (unit == 1.0) return max
        val transformed = (1.0 - unit) * feature(min) + unit * feature(max)
        return when (scale) {
            Scale.Linear -> transformed
            Scale.Log -> exp(transformed)
            Scale.ReverseLog -> max - kotlin.math.expm1(-transformed)
        }.coerceIn(min, max)
    }
}

internal fun DecisionSpaceDef.numericDomains(prefix: String = ""): Map<String, NumericDomain> = buildMap {
    for ((name, spec) in variables + optionalVariables + context + optionalContext) {
        val scale = scales[name] ?: Scale.Linear
        val domain = when (spec) {
            is FloatSpec -> NumericDomain(spec.min, spec.max, scale)
            is IntSpec -> NumericDomain(spec.min.toDouble(), spec.max.toDouble(), scale)
            else -> null
        }
        if (domain != null) put(prefix + name, domain)
    }
    for ((name, child) in spaces + optionalSpaces) putAll(child.numericDomains("$prefix$name."))
}
