package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.BoolContextHandle
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.FeatureEncoder
import com.eignex.combo.decisions.IntContextHandle
import com.eignex.combo.shared.LinearFeatureProjection
import com.eignex.combo.shared.WitnessValidator
import com.eignex.combo.shared.nominalSide
import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.IntHandle
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.Sample
import com.eignex.koblas.DenseVector
import com.eignex.koblas.Vector
import kotlin.math.roundToLong

/**
 * BO feature encoding with numeric base coordinates normalized to [0, 1] in the declared geometry.
 * Boolean presence and categorical indicators retain their solver slots. Inactive values are masked
 * before normalization; interactions are recomputed from normalized scalars. Solver values are unchanged.
 *
 * @property space Original decision space whose coordinate order defines the encoding.
 */
class SearchTransform(override val space: CompiledDecisionSpace) : FeatureEncoder<Vector> {
    private val validator = WitnessValidator(space)
    private val base = LinearFeatureProjection(space)
    private val layout = base.layout
    override val featureSize: Int get() = base.featureSize

    private val slots = space.numericDomains.mapNotNull { (name, domain) ->
        val intId = space.compiled.intVarIdByName[name]
        val realId = layout.realHandles.entries.firstOrNull { it.value.name == name }?.key
        val slot = if (intId != null) layout.intStart + intId else realId?.let { layout.realStart + it }
        slot?.let { Triple(name, it, domain) }
    }

    /**
     * Decode normalized base coordinates against an anchor, returning only a checked feasible assignment.
     * Categorical indicators must be discrete. Derived interaction slots are ignored and recomputed by encode.
     * Auxiliary solver coordinates absent from the declared numeric domains retain the anchor's values.
     */
    fun decode(coordinates: Vector, anchor: BanditSample, context: Context = Context.Empty): BanditSample? {
        require(coordinates.size == featureSize && coordinates.toDoubleArray().all { it.isFinite() })
        val bools = anchor.sample.bools
        val ints = anchor.sample.ints
        val reals = anchor.sample.reals
        for (id in bools.indices) {
            val value = coordinates[layout.boolStart + id]
            if (value != 0.0 && value != 1.0) return null
            bools[id] = value == 1.0
        }
        for ((name, slot, domain) in slots) {
            val unit = coordinates[slot]
            if (unit !in 0.0..1.0) return null
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
                reals[id] = value
            }
        }
        val sample = Sample(bools, ints, reals, exactReals = reals.map { BigFraction.ofDouble(it)!! })
        return sample.takeIf {
            space.matches(
                it,
                space.assumptionsFor(context),
            ) && validator.valid(it)
        }?.let(::BanditSample)
    }

    override fun encode(sample: BanditSample, context: Context): Vector {
        val features = base.encode(sample, context).toDoubleArray()
        for ((name, slot, domain) in slots) {
            val active = space.activeConditions[name]?.let { space.evaluate(it, sample.sample) } ?: true
            if (!active) continue
            val lo = domain.feature(domain.min)
            val hi = domain.feature(domain.max)
            features[slot] = if (lo == hi) 0.0 else (features[slot] - lo) / (hi - lo)
        }
        fun scalar(handle: Any): Double = when (handle) {
            is BoolHandle -> features[layout.boolStart + space.compiled.boolVarIdByName.getValue(handle.name)]
            is IntHandle -> features[layout.intStart + space.compiled.intVarIdByName.getValue(handle.name)]
            is BoolContextHandle -> scalar(handle.klauseHandle)
            is IntContextHandle -> scalar(handle.klauseHandle)
            else -> error("unsupported interaction scalar")
        }
        for (interaction in layout.interactions) {
            val start = layout.interactionStart.getValue(interaction)
            val nominal = interaction.nominalSide()
            if (nominal == null) {
                features[start] = scalar(interaction.lhs) * scalar(interaction.rhs)
            } else {
                val factor = scalar(if (interaction.lhs === nominal) interaction.rhs else interaction.lhs)
                val indicators = space.compiled.nominalIndicators.getValue(nominal.name)
                nominal.labels.forEachIndexed { index, label ->
                    features[start + index] = if (sample.sample.boolValue(indicators.getValue(label))) factor else 0.0
                }
            }
        }
        return DenseVector.wrap(features)
    }
}
