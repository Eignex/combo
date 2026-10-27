package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.FeatureEncoder
import com.eignex.combo.shared.SolverCoordinateLayout
import com.eignex.combo.shared.WitnessValidator
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.Sample
import com.eignex.koblas.DenseVector
import com.eignex.koblas.Vector
import kotlin.math.roundToLong

/**
 * BO feature encoding with numeric base coordinates normalized to [0, 1] in the declared geometry.
 * Boolean presence and categorical indicators retain their solver slots. Inactive values are masked
 * before normalization. There are no derived model features: use [InteractionFeatureProjection]
 * when a model needs declared interaction products. Solver values are unchanged.
 *
 * @property space Original decision space whose coordinate order defines the encoding.
 */
class SearchTransform(override val space: CompiledDecisionSpace) : FeatureEncoder<Vector> {
    private val validator = WitnessValidator(space)
    internal val layout = SolverCoordinateLayout(space)
    override val featureSize: Int get() = layout.size

    private val boolNames = space.compiled.boolVarIdByName.entries.associate { it.value to it.key } +
        space.compiled.nominalIndicators.flatMap { (name, indicators) -> indicators.values.map { it to name } }.toMap()
    private val intNames = space.compiled.intVarIdByName.entries.associate { it.value to it.key }
    private fun active(name: String?, sample: Sample): Boolean =
        space.activeConditions[name]?.let { space.evaluate(it, sample) } ?: true

    private val slots = space.numericDomains.mapNotNull { (name, domain) ->
        val intId = space.compiled.intVarIdByName[name]
        val realId = layout.realHandles.entries.firstOrNull { it.value.name == name }?.key
        val slot = if (intId != null) layout.intStart + intId else realId?.let { layout.realStart + it }
        slot?.let { Triple(name, it, domain) }
    }

    /** Number of nonconstant independent decision variables, excluding pinned contexts and category indicators. */
    val dimension: Int get() {
        val contexts = (
            space.contextBools.map { it.name } + space.contextInts.map { it.name } +
                space.contextFloats.map { it.name } +
                space.contextBools.mapNotNull { it.isKnownGate?.name } +
                space.contextInts.mapNotNull { it.isKnownGate?.name }
            ).toSet()
        val indicators = space.compiled.nominalIndicators.values.flatMap { it.values }.toSet()
        return space.numericDomains.count { (name, domain) -> name !in contexts && domain.min < domain.max } +
            space.compiled.boolVarIdByName.count { (name, id) -> name !in contexts && id !in indicators } +
            space.compiled.nominalIndicators.count { it.value.size > 1 }
    }

    /**
     * Map continuous unit draws to primitive decisions, retaining hard context and canonical inactive values.
     * Categories select one label; constraints are checked in original coordinates. A rejected proposal is null.
     */
    fun sampleUnit(units: Vector, anchor: BanditSample, context: Context = Context.Empty): BanditSample? {
        require(units.size == featureSize && units.toDoubleArray().all { it >= 0.0 && it < 1.0 })
        val assumptions = space.discreteAssumptionsFor(context)
        val bools = anchor.sample.bools
        val ints = anchor.sample.ints
        val reals = anchor.sample.reals
        val exact = (anchor.sample.exactReals ?: reals.map { BigFraction.ofDouble(it)!! }).toMutableList()
        for (id in bools.indices) bools[id] = assumptions.bools[id] ?: (units[id] >= 0.5)
        for ((_, indicators) in space.compiled.nominalIndicators) {
            val ids = indicators.values.toList()
            val chosen = (units[ids.first()] * ids.size).toInt()
            ids.forEachIndexed { index, id -> bools[id] = index == chosen }
        }
        val activation = Sample(bools, ints, reals, exactReals = exact)
        for ((id, name) in boolNames) {
            if (!active(name, activation) && name !in space.compiled.nominalIndicators) bools[id] = false
        }
        for ((name, indicators) in space.compiled.nominalIndicators) {
            if (!active(name, activation)) indicators.values.forEachIndexed { index, id -> bools[id] = index == 0 }
        }
        for ((name, slot, domain) in slots) {
            if (space.contextFloats.any { it.name == name }) continue
            val intId = space.compiled.intVarIdByName[name]
            if (intId != null && intId in assumptions.ints) {
                ints[intId] = assumptions.ints.getValue(intId)
                continue
            }
            val value = domain.fromUnit(if (active(name, activation)) units[slot] else 0.0)
            if (intId != null) {
                val spec = layout.floatSpecs[intId]
                ints[intId] = if (spec == null) {
                    value.roundToLong()
                } else if (spec.scale == 0.0) {
                    0L
                } else {
                    ((value - spec.min) / spec.scale).roundToLong()
                }
            } else {
                val id = layout.realHandles.entries.first { it.value.name == name }.key
                if (value == reals[id]) continue
                exact[id] = BigFraction.ofDouble(value)!!
                reals[id] = value
            }
        }
        val proposed = Sample(bools, ints, reals, exactReals = exact)
        return proposed.takeIf { space.matches(it, context) && validator.valid(it) }?.let(::BanditSample)
    }

    /**
     * Decode normalized base coordinates against an anchor, returning only a checked feasible assignment.
     * Categorical indicators must be discrete; this accepts base coordinates only.
     * Auxiliary solver coordinates absent from the declared numeric domains retain the anchor's values.
     */
    fun decode(coordinates: Vector, anchor: BanditSample, context: Context = Context.Empty): BanditSample? {
        require(coordinates.size == featureSize && coordinates.toDoubleArray().all { it.isFinite() })
        val bools = anchor.sample.bools
        val ints = anchor.sample.ints
        val reals = anchor.sample.reals
        val exact = (anchor.sample.exactReals ?: reals.map { BigFraction.ofDouble(it)!! }).toMutableList()
        val original = encode(anchor, context)
        for (id in bools.indices) {
            val value = coordinates[layout.boolStart + id]
            if (value != 0.0 && value != 1.0) return null
            if (value == original[layout.boolStart + id]) continue
            bools[id] = value == 1.0
        }
        for ((name, slot, domain) in slots) {
            val unit = coordinates[slot]
            if (unit !in 0.0..1.0) return null
            // Preserve exact witnesses and avoid roundoff from inverting unchanged scaled coordinates.
            if (unit == original[slot]) continue
            val value = domain.fromUnit(unit)
            val intId = space.compiled.intVarIdByName[name]
            if (intId != null) {
                val spec = layout.floatSpecs[intId]
                ints[intId] = if (spec == null) {
                    value.roundToLong()
                } else if (spec.scale == 0.0) {
                    0L
                } else {
                    ((value - spec.min) / spec.scale).roundToLong()
                }
            } else {
                val id = layout.realHandles.entries.first { it.value.name == name }.key
                if (value != reals[id]) exact[id] = BigFraction.ofDouble(value)!!
                reals[id] = value
            }
        }
        val sample = Sample(bools, ints, reals, exactReals = exact)
        return sample.takeIf {
            space.matches(
                it,
                context,
            ) && validator.valid(it)
        }?.let(::BanditSample)
    }

    override fun encode(sample: BanditSample, context: Context): Vector {
        val source = sample.sample
        val features = DoubleArray(featureSize)
        for (id in 0 until layout.numBoolVars) {
            if (active(boolNames[id], source)) features[id] = if (source.boolValue(id)) 1.0 else 0.0
        }
        for (id in 0 until layout.numIntVars) {
            if (active(intNames[id], source)) {
                val raw = source.intValue(id)
                features[layout.intStart + id] = layout.floatSpecs[id]?.realValue(Math.toIntExact(raw))
                    ?: raw.toDouble()
            }
        }
        for ((id, handle) in layout.realHandles) {
            if (active(handle.name, source)) features[layout.realStart + id] = sample.float(handle, space)
        }
        for ((name, slot, domain) in slots) {
            if (!active(name, source)) continue
            val lo = domain.feature(domain.min)
            val hi = domain.feature(domain.max)
            features[slot] = if (lo == hi) 0.0 else (domain.feature(features[slot]) - lo) / (hi - lo)
        }
        return DenseVector.wrap(features)
    }
}
