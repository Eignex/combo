package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.BoolContextHandle
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.FeatureEncoder
import com.eignex.combo.decisions.IntContextHandle
import com.eignex.combo.shared.nominalSide
import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.IntHandle
import com.eignex.koblas.DenseVector
import com.eignex.koblas.Vector

/**
 * Optional model features: normalized base coordinates followed by declared interaction products.
 * Forests normally use [SearchTransform] alone. This adapter never changes proposal geometry;
 * decoding and feasibility always operate on base coordinates and original solver assignments.
 *
 * @property coordinates Independent base-coordinate transform.
 */
class InteractionFeatureProjection(val coordinates: SearchTransform) : FeatureEncoder<Vector> {
    override val space get() = coordinates.space
    override val featureSize =
        coordinates.featureSize + space.interactions.sumOf { it.nominalSide()?.labels?.size ?: 1 }

    override fun encode(sample: BanditSample, context: Context): Vector {
        val base = coordinates.encode(sample, context)
        val features = base.toDoubleArray().copyOf(featureSize)
        val layout = coordinates.layout
        fun scalar(handle: Any): Double = when (handle) {
            is BoolHandle -> base[layout.boolStart + space.coordinates.boolVarIdByName.getValue(handle.name)]
            is IntHandle -> base[layout.intStart + space.coordinates.intVarIdByName.getValue(handle.name)]
            is BoolContextHandle -> scalar(handle.klauseHandle)
            is IntContextHandle -> scalar(handle.klauseHandle)
            else -> error("unsupported interaction scalar")
        }
        var slot = coordinates.featureSize
        for (interaction in space.interactions) {
            val nominal = interaction.nominalSide()
            if (nominal == null) {
                features[slot++] = scalar(interaction.lhs) * scalar(interaction.rhs)
            } else {
                val factor = scalar(if (interaction.lhs === nominal) interaction.rhs else interaction.lhs)
                val indicators = space.coordinates.nominalIndicators.getValue(nominal.name)
                for (label in nominal.labels) {
                    val id = indicators.getValue(label)
                    features[slot++] = factor * base[layout.boolStart + id]
                }
            }
        }
        return DenseVector.wrap(features)
    }
}
