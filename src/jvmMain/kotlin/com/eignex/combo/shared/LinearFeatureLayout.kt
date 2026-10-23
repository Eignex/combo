package com.eignex.combo.shared

import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.InteractionHandle
import com.eignex.klause.model.FloatSpec
import com.eignex.klause.schema.FloatHandle
import com.eignex.klause.schema.NominalHandle

/**
 * Feature index for Klause variables and declared interactions.
 *
 * Layout: `[Boolean coordinates | integer coordinates | real coordinates | interactions]`.
 * Contexts are ordinary solver variables pinned through assumptions for each call.
 * [realHandles] identifies native real coordinates. If Klause selects a finite float
 * representation, [floatSpecs] decodes its integer coordinates into the variable's
 * declared units. Combo does not select a representation or alter solver values.
 * [LinearFeatureProjection] uses the same mapping to construct optimization objectives.
 */
class LinearFeatureLayout internal constructor(
    val numBoolVars: Int,
    val numIntVars: Int,
    /** Number of continuous solver coordinates. */
    val numRealVars: Int,
    /** Float handles indexed by their continuous solver coordinate. */
    val realHandles: Map<Int, FloatHandle>,
    val interactions: List<InteractionHandle>,
    val floatSpecs: Map<Int, FloatSpec>,
) {
    val boolStart: Int = 0
    val intStart: Int = numBoolVars

    /** First continuous feature slot, following the Boolean and integer blocks. */
    val realStart: Int = numBoolVars + numIntVars
    val interactionsStart: Int = realStart + numRealVars

    internal val interactionSlotCount: Map<InteractionHandle, Int> =
        interactions.associateWith { it.nominalSide()?.labels?.size ?: 1 }

    internal val interactionStart: Map<InteractionHandle, Int> = run {
        val out = mutableMapOf<InteractionHandle, Int>()
        var cursor = interactionsStart
        for (h in interactions) {
            out[h] = cursor
            cursor += interactionSlotCount.getValue(h)
        }
        out
    }

    val featureSize: Int = interactionsStart + interactionSlotCount.values.sum()

    /** Decode a bucket-int back into its real-value Double for int slot [intVarId].
     *  Identity (`bucket.toDouble()`) when the slot is a true int. */
    internal fun realValue(intVarId: Int, bucket: Long): Double =
        floatSpecs[intVarId]?.realValue(Math.toIntExact(bucket)) ?: bucket.toDouble()

    /**
     * Translate a model weight (per-unit-real-value) for int slot [intVarId] into the
     * pair `(bucketCoeff, constantContribution)`. For true ints the pair is `(w, 0)`;
     * for floats it's `(w · scale, w · min)`. The caller accumulates the constant
     * contribution into the objective's bias term.
     */
    internal fun coefficientForInt(intVarId: Int, modelWeight: Double): Pair<Double, Double> {
        val spec = floatSpecs[intVarId] ?: return modelWeight to 0.0
        return modelWeight * spec.scale to modelWeight * spec.min
    }

    companion object {
        fun from(space: CompiledDecisionSpace): LinearFeatureLayout {
            val floatSpecs = mutableMapOf<Int, FloatSpec>()
            for ((name, spec) in space.compiled.floatDecoders) {
                val id = space.compiled.intVarIdByName[name] ?: continue
                floatSpecs[id] = spec
            }
            return LinearFeatureLayout(
                numBoolVars = space.compiled.problem.numBoolVars,
                numIntVars = space.compiled.problem.numIntVars,
                numRealVars = space.compiled.problem.numRealVars,
                realHandles = space.compiled.floatDecoders.mapNotNull { (name, spec) ->
                    if (name in space.compiled.intVarIdByName) return@mapNotNull null
                    val handle = FloatHandle(name, spec.min, spec.max)
                    val id = space.compiled.minimize(handle).realCoefficients.indexOfFirst { it != 0.0 }
                    check(id >= 0) { "float '$name' has no solver coordinate" }
                    id to handle
                }.toMap(),
                interactions = space.interactions,
                floatSpecs = floatSpecs,
            )
        }
    }
}

internal fun InteractionHandle.nominalSide(): NominalHandle? = when {
    lhs is NominalHandle -> lhs
    rhs is NominalHandle -> rhs
    else -> null
}
