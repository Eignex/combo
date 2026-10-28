package com.eignex.combo.bandit.dt

import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.klause.util.bigIntOf
import com.eignex.klause.util.minus
import com.eignex.klause.util.plus
import com.eignex.klause.util.times
import com.eignex.klause.util.div
import com.eignex.klause.util.toLongExact
import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.FloatHandle
import com.eignex.klause.schema.IntHandle
import com.eignex.klause.schema.NominalHandle

/**
 * Default candidate-[Split] generator for a [DecisionTreeBandit] over [space]. Emits:
 *
 *  - One [BoolSplit] per "true" bool variable. Nominal indicator bits and the
 *    auto-allocated isKnown_* gates of optional variables both live in
 *    `boolVarIdByName` too — gates are kept (splitting on presence is useful);
 *    indicator bits are dropped in favor of [NominalSplit]s, which carry the
 *    same routing information in a typed form.
 *  - One one-vs-rest [NominalSplit] per (nominal, label) pair. Multi-label
 *    partitions aren't enumerated by default — too many candidates without
 *    user guidance.
 *  - [numericThresholds] [IntThresholdSplit]s per int variable, at equally-spaced
 *    thresholds across the domain's declared search geometry. Bucketed floats are *not* emitted here —
 *    they get [FloatThresholdSplit]s instead (see below).
 *  - [numericThresholds] [FloatThresholdSplit]s per float variable, at equally-
 *    spaced thresholds in the declared search geometry, decoded into `[min, max]` — independent
 *    of klause's bucket count. Tree branching semantics therefore don't shift
 *    when the user changes the bucket count.
 *
 * Callers can pass their own list to the bandit — this is just the no-argument
 * default that "does something reasonable" for an unfamiliar space.
 */
fun defaultSplitCandidates(space: CompiledDecisionSpace, numericThresholds: Int = 4): List<Split> {
    val out = mutableListOf<Split>()
    val compiled = space.coordinates

    val indicatorBoolIds = compiled.nominalIndicators.values.flatMap { it.values }.toSet()
    val floatNames = compiled.floatDecoders.keys

    // True bool variables (excluding nominal indicator bits, but keeping isKnown_* gates).
    for ((name, id) in compiled.boolVarIdByName) {
        if (id in indicatorBoolIds) continue
        out += BoolSplit(BoolHandle(name))
    }

    // True int variables (excluding the bucketed-int representation of float vars).
    for ((name, _) in compiled.intVarIdByName) {
        if (name in floatNames) continue
        val handle = intHandleFor(space, name) ?: continue
        if (compiled.schema == null) {
            if (numericThresholds <= 0 || handle.min == handle.max) continue
            val lower = bigIntOf(handle.min)
            val span = bigIntOf(handle.max) - lower
            out += List(numericThresholds) { i ->
                val offset = span * bigIntOf(i.toLong() + 1) / bigIntOf(numericThresholds.toLong() + 1)
                IntThresholdSplit(handle, (lower + offset).toLongExact())
            }.distinctBy { it.threshold }
            continue
        }
        out += thresholdsAcross(space, name, handle.min.toDouble(), handle.max.toDouble(), numericThresholds).map { t ->
            IntThresholdSplit(handle, t.toLong())
        }
    }

    // Nominal variables: one-vs-rest per label.
    for ((name, indicators) in compiled.nominalIndicators) {
        val labels = indicators.keys.toList()
        if (labels.size < 2) continue
        val handle = NominalHandle(name, labels)
        for (label in labels) {
            out += NominalSplit(handle, label = label)
        }
    }

    // Float variables: thresholds in **real-value units**.
    for ((name, spec) in compiled.floatDecoders) {
        val handle = FloatHandle(name, spec.min, spec.max)
        out += thresholdsAcross(space, name, handle.min, handle.max, numericThresholds).map { t ->
            FloatThresholdSplit(handle, t)
        }
    }

    if (compiled.schema == null) for ((_, handle) in compiled.realHandles) {
        if (!handle.min.isFinite() || !handle.max.isFinite()) continue
        out += thresholdsAcross(space, handle.name, handle.min, handle.max, numericThresholds).map {
            FloatThresholdSplit(handle, it)
        }
    }
    return out
}

/**
 * Build [k] internal thresholds across `(lo, hi)` — equally spaced, excluding both
 * endpoints, respecting any declared scale hint. Returns an empty list when no strict interior threshold fits (e.g.,
 * `hi - lo <= 0`).
 */
private fun thresholdsAcross(space: CompiledDecisionSpace, name: String, lo: Double, hi: Double, k: Int): List<Double> {
    val span = hi - lo
    if (span <= 0.0 || k <= 0) return emptyList()
    return List(k) { i ->
        val unit = (i + 1).toDouble() / (k + 1)
        space.numericDomains[name]?.fromUnit(unit) ?: (lo + unit * span)
    }
}

/**
 * Build [IntHandle] for an int variable named [name] in [space]. Returns null when the
 * domain min/max isn't recoverable from the decision-space definition (e.g., the int
 * was added via raw klause and not declared with a domain).
 */
private fun intHandleFor(space: CompiledDecisionSpace, name: String): IntHandle? {
    if (space.coordinates.schema == null) {
        val id = space.coordinates.intVarIdByName[name] ?: return null
        val bounds = space.problem.intBounds
        if (!bounds.hasLower(id) || !bounds.hasUpper(id)) return null
        return IntHandle(name, bounds.lower(id), bounds.upper(id))
    }
    val spec = lookupIntSpec(space.definition, name) ?: return null
    return IntHandle(name, spec.min.toLong(), spec.max.toLong())
}

/**
 * Walk the (hierarchical) [com.eignex.combo.decisions.DecisionSpaceDef] looking for an
 * [com.eignex.klause.model.IntSpec] keyed by the fully-qualified [name]. The compiler
 * qualifies names with dotted prefixes, so we mirror that walk here.
 */
private fun lookupIntSpec(
    def: com.eignex.combo.decisions.DecisionSpaceDef,
    name: String,
    prefix: String = "",
): com.eignex.klause.model.IntSpec? {
    for ((local, spec) in def.variables) {
        if (prefix + local == name && spec is com.eignex.klause.model.IntSpec) return spec
    }
    for ((local, spec) in def.optionalVariables) {
        if (prefix + local == name && spec is com.eignex.klause.model.IntSpec) return spec
    }
    for ((local, spec) in def.context) {
        if (prefix + local == name && spec is com.eignex.klause.model.IntSpec) return spec
    }
    for ((local, spec) in def.optionalContext) {
        if (prefix + local == name && spec is com.eignex.klause.model.IntSpec) return spec
    }
    for ((local, child) in def.spaces) {
        lookupIntSpec(child, name, prefix = "$prefix$local.")?.let { return it }
    }
    for ((local, child) in def.optionalSpaces) {
        lookupIntSpec(child, name, prefix = "$prefix$local.")?.let { return it }
    }
    return null
}
