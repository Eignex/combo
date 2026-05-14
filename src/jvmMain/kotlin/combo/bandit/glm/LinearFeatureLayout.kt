package combo.bandit.glm

import com.eignex.klause.schema.NominalHandle
import combo.decisions.CompiledDecisionSpace
import combo.decisions.InteractionHandle

/**
 * Linear-bandit-specific feature index over a [CompiledDecisionSpace]. Maps every
 * solver variable (decisions and contexts alike) plus declared interactions to a stable
 * slot in the dense weight vector that the linear models train against.
 *
 * Layout: `[all bool vars | all int vars | interactions]`
 *
 * Contexts no longer get their own block — they are solver variables now, pinned per
 * call via [com.eignex.klause.solver.Assumptions]. Interactions involving a nominal
 * decision expand to one slot per label.
 *
 * **Float scaling.** Klause represents a float variable as a bucketed int internally,
 * but the bandit fits its model in real-value (Double) units — credible intervals,
 * EI/UCB calculations, and posterior draws are all dimensionally meaningful only when
 * the linear weight is per-unit-real-value. [floatScaling] records the affine map
 * `real = scale · bucket + offset` for every int slot that is actually a bucketed
 * float, where:
 *   - `scale  = (max - min) / (buckets - 1)`
 *   - `offset = min`
 *
 * [LinearFeatureProjection] uses this to:
 *   - emit the *decoded* real value at the int slot in [encode], so models see real-
 *     value features;
 *   - rewrite a model weight `w` (per-unit-real-value) into the bucket coefficient
 *     `w · scale` for klause's [LinearObjective], while accumulating `w · offset` into
 *     the objective's constant term so the round-trip is exact.
 *
 * True int variables (no entry in [floatScaling]) keep the identity scaling — the
 * feature is the int value, the weight rolls straight into klause's coefficient.
 */
class LinearFeatureLayout internal constructor(
    val numBoolVars: Int,
    val numIntVars: Int,
    val interactions: List<InteractionHandle>,
    val floatScaling: Map<Int, FloatScaling>,
) {
    val boolStart: Int = 0
    val intStart: Int = numBoolVars
    val interactionsStart: Int = numBoolVars + numIntVars

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
    internal fun realValue(intVarId: Int, bucket: Int): Double {
        val s = floatScaling[intVarId] ?: return bucket.toDouble()
        return s.scale * bucket + s.offset
    }

    /**
     * Translate a model weight (per-unit-real-value) for int slot [intVarId] into the
     * pair `(bucketCoeff, constantContribution)`. For true ints the pair is `(w, 0)`;
     * for floats it's `(w · scale, w · offset)`. The caller accumulates the constant
     * contribution into the objective's bias term.
     */
    internal fun coefficientForInt(intVarId: Int, modelWeight: Double): Pair<Double, Double> {
        val s = floatScaling[intVarId] ?: return modelWeight to 0.0
        return modelWeight * s.scale to modelWeight * s.offset
    }

    companion object {
        fun from(space: CompiledDecisionSpace): LinearFeatureLayout {
            val floatScaling = mutableMapOf<Int, FloatScaling>()
            for ((name, spec) in space.compiled.floatDecoders) {
                val id = space.compiled.intVarIdByName[name] ?: continue
                val span = spec.max - spec.min
                val divisor = (spec.buckets - 1).coerceAtLeast(1).toDouble()
                floatScaling[id] = FloatScaling(
                    scale = span / divisor,
                    offset = spec.min,
                )
            }
            return LinearFeatureLayout(
                numBoolVars = space.compiled.problem.numBoolVars,
                numIntVars = space.compiled.problem.numIntVars,
                interactions = space.interactions,
                floatScaling = floatScaling,
            )
        }
    }
}

/** Affine map between a bucketed-int and its real-value Double: `real = scale · b + offset`. */
data class FloatScaling(val scale: Double, val offset: Double)

internal fun InteractionHandle.nominalSide(): NominalHandle? = when {
    lhs is NominalHandle -> lhs
    rhs is NominalHandle -> rhs
    else -> null
}
