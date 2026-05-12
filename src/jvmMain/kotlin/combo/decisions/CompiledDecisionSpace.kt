package combo.decisions

import com.eignex.klause.ast.SchemaEntry
import com.eignex.klause.compile.CompiledProblem
import com.eignex.klause.solver.LinearObjective
import com.eignex.klause.solver.Sample
import com.eignex.skema.SchemaDef
import combo.math.Vector
import combo.math.VectorView
import combo.math.vectors

/**
 * Compiled [DecisionSpace]: a klause [CompiledProblem] (decision side) plus a
 * [FeatureLayout] describing the dense feature vector that linear bandits work in.
 *
 * Use [encode] to project a (`Sample`, `Context`) pair into a feature vector and
 * [toObjective] to translate a learned weight vector back into a klause
 * [LinearObjective] whose minimiser is the best-scoring feasible `Sample`.
 */
class CompiledDecisionSpace internal constructor(
    val compiled: CompiledProblem,
    val layout: FeatureLayout,
    val schemaDef: SchemaDef<SchemaEntry>,
) {
    val featureSize: Int get() = layout.featureSize

    /**
     * Project (`sample`, `context`) into a fresh dense feature vector.
     * Bool features are 0/1; int features are raw ints; context features are pulled
     * from the [Context]. Missing context values throw via [Context.get].
     */
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
     * Translate a learned weight vector into a klause [LinearObjective]. Context-feature
     * contributions become part of the objective's `constant`; only decision-feature
     * weights drive klause's search.
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
        // Context contributions are constant during this choose call.
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
