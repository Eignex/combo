package combo.bandit.glm

import com.eignex.klause.ast.BoolExpr
import com.eignex.klause.solver.LinearObjective
import com.eignex.klause.solver.Sample
import combo.decisions.CompiledDecisionSpace
import combo.decisions.Context
import combo.decisions.FeatureEncoder
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
 *
 * Optional decision variables: when their activation condition evaluates to false on
 * a sample, the corresponding feature slot is zeroed. The wrapped [LinearModel]'s
 * iterator yields all indices, but with x_i = 0 the gradient contribution is exactly
 * zero — equivalent to "skip update". Klause's pinning constraint keeps the value at
 * its default for inactive samples, so klause's search-time objective stays consistent
 * for optional bool slots (whose default is 0). For optional int slots with non-zero
 * `min`, there is a small bias at choose-time equal to `w_i * min` on inactive
 * assignments; this is acceptable for the common case `min == 0` and documented for
 * the rare case.
 */
class LinearFeatureProjection(override val space: CompiledDecisionSpace) : FeatureEncoder<VectorView> {

    val layout: LinearFeatureLayout = LinearFeatureLayout.from(space)
    override val featureSize: Int get() = layout.featureSize

    // Precompute (klause-id → active condition) for the optional slots only, so the
    // common path (no optionals) pays nothing.
    private val optionalBoolIds: Map<Int, BoolExpr> = run {
        val out = mutableMapOf<Int, BoolExpr>()
        for ((name, id) in space.compiled.boolVarIdByName) {
            space.activeConditions[name]?.let { out[id] = it }
        }
        out
    }
    private val optionalIntIds: Map<Int, BoolExpr> = run {
        val out = mutableMapOf<Int, BoolExpr>()
        for ((name, id) in space.compiled.intVarIdByName) {
            space.activeConditions[name]?.let { out[id] = it }
        }
        out
    }

    override fun encode(sample: Sample, context: Context): Vector {
        val out = vectors.zeroVector(layout.featureSize)
        for (b in 0 until layout.numBoolDecisions) {
            val cond = optionalBoolIds[b]
            val active = cond == null || space.isActive(klauseBoolHandleFor(b), sample)
            if (active) {
                out[layout.boolDecisionsStart + b] = if (sample.bools[b]) 1f else 0f
            }
        }
        for (i in 0 until layout.numIntDecisions) {
            val cond = optionalIntIds[i]
            val active = cond == null || space.isActive(klauseIntHandleFor(i), sample)
            if (active) {
                out[layout.intDecisionsStart + i] = sample.ints[i].toFloat()
            }
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

    // Lookups by klause id back to the named handles, used only on the optional path.
    private val boolNameById: Array<String?> = arrayOfNulls<String>(space.compiled.problem.numBoolVars).also { arr ->
        for ((name, id) in space.compiled.boolVarIdByName) arr[id] = name
    }
    private val intNameById: Array<String?> = arrayOfNulls<String>(space.compiled.problem.numIntVars).also { arr ->
        for ((name, id) in space.compiled.intVarIdByName) arr[id] = name
    }

    private fun klauseBoolHandleFor(id: Int): com.eignex.klause.schema.BoolHandle =
        com.eignex.klause.schema.BoolHandle(boolNameById[id]!!)

    private fun klauseIntHandleFor(id: Int): com.eignex.klause.schema.IntHandle {
        val name = intNameById[id]!!
        // Recover (min, max) from the spec carried by the compiled problem.
        val domain = space.compiled.problem.intDomains[id]
        return com.eignex.klause.schema.IntHandle(name, domain.min, domain.max)
    }
}
