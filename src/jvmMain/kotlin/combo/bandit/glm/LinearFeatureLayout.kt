package combo.bandit.glm

import com.eignex.klause.ast.FloatSpec
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
 * the linear weight is per-unit-real-value. [floatSpecs] records the klause [FloatSpec]
 * for every int slot that is actually a bucketed float; the bucket↔real affine map
 * (`real = scale · bucket + min`) is read straight off [FloatSpec.scale] /
 * [FloatSpec.realValue], the single source of truth klause's own decode uses.
 *
 * [LinearFeatureProjection] uses this to:
 *   - emit the *decoded* real value at the int slot in [encode], so models see real-
 *     value features;
 *   - rewrite a model weight `w` (per-unit-real-value) into the bucket coefficient
 *     `w · scale` for klause's [LinearObjective], while accumulating `w · min` into
 *     the objective's constant term so the round-trip is exact.
 *
 * True int variables (no entry in [floatSpecs]) keep the identity scaling — the
 * feature is the int value, the weight rolls straight into klause's coefficient.
 */
class LinearFeatureLayout internal constructor(
    val numBoolVars: Int,
    val numIntVars: Int,
    val interactions: List<InteractionHandle>,
    val floatSpecs: Map<Int, FloatSpec>,
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
    internal fun realValue(intVarId: Int, bucket: Int): Double =
        floatSpecs[intVarId]?.realValue(bucket) ?: bucket.toDouble()

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
