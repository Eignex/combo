package combo.bandit.glm

import com.eignex.klause.solver.LinearObjective
import com.eignex.klause.solver.Sample
import combo.decisions.CompiledDecisionSpace
import combo.decisions.Context
import combo.math.Vector
import combo.math.VectorView
import combo.math.vectors

/**
 * Linear bandit's view of a [CompiledDecisionSpace]: projects `(Sample, Context)` into
 * a dense feature vector, and translates a learned weight vector back into a klause
 * [LinearObjective] for choose-time optimisation.
 *
 * Bool features are 0/1, int features are raw ints, context features are pulled from
 * the [Context]. Missing context values throw via [Context.get].
 */
class LinearFeatureProjection(val space: CompiledDecisionSpace) {

    val layout: LinearFeatureLayout = LinearFeatureLayout.from(space)
    val featureSize: Int get() = layout.featureSize

    fun encode(sample: Sample, context: Context = Context.Empty): Vector {
        val out = vectors.zeroVector(layout.featureSize)
        for (b in 0 until layout.numBoolDecisions) {
            out[layout.boolDecisionsStart + b] = if (sample.bools[b]) 1f else 0f
        }
        for (i in 0 until layout.numIntDecisions) {
            out[layout.intDecisionsStart + i] = sample.ints[i].toFloat()
        }
        for ((handle, idx) in layout.boolContextIndex) {
            out[idx] = if (context[handle]) 1f else 0f
        }
        for ((handle, idx) in layout.intContextIndex) {
            out[idx] = context[handle].toFloat()
        }
        return out
    }

    /**
     * Build a klause objective whose minimiser is the best-scoring feasible [Sample]
     * under [weights] given [context]. Context-feature contributions are folded into
     * the objective's `constant` (they're invariant during this choose). `maximize=true`
     * flips signs so klause's minimiser searches for the bandit's argmax.
     */
    fun toObjective(weights: VectorView, bias: Float, context: Context, maximize: Boolean): LinearObjective {
        require(weights.size == layout.featureSize) {
            "weights size ${weights.size} must match feature layout ${layout.featureSize}"
        }
        val sign = if (maximize) -1.0 else 1.0
        val boolWeights = DoubleArray(layout.numBoolDecisions) {
            sign * weights[layout.boolDecisionsStart + it].toDouble()
        }
        val intCoefficients = DoubleArray(layout.numIntDecisions) {
            sign * weights[layout.intDecisionsStart + it].toDouble()
        }
        var constant = bias.toDouble()
        for ((handle, idx) in layout.boolContextIndex) {
            if (context[handle]) constant += weights[idx].toDouble()
        }
        for ((handle, idx) in layout.intContextIndex) {
            constant += weights[idx].toDouble() * context[handle]
        }
        return LinearObjective(boolWeights, intCoefficients, constant = sign * constant)
    }
}
