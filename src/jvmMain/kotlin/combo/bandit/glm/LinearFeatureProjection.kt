package combo.bandit.glm

import com.eignex.klause.ast.BoolExpr
import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.IntHandle
import com.eignex.klause.solver.LinearObjective
import com.eignex.klause.solver.Sample
import combo.decisions.BoolContextHandle
import combo.decisions.CompiledDecisionSpace
import combo.decisions.Context
import combo.decisions.FeatureEncoder
import combo.decisions.IntContextHandle
import combo.decisions.InteractionHandle
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
 * assignments; acceptable for the common case `min == 0`.
 *
 * Interactions (declared via [combo.decisions.DecisionSpace.interact]) get an extra
 * feature slot. `context × decision` interactions fold into the bool/int coefficients
 * of klause's objective at choose-time; `context × context` interactions become a
 * constant.
 */
class LinearFeatureProjection(override val space: CompiledDecisionSpace) : FeatureEncoder<VectorView> {

    val layout: LinearFeatureLayout = LinearFeatureLayout.from(space)
    override val featureSize: Int get() = layout.featureSize

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

    private val boolNameById: Array<String?> = arrayOfNulls<String>(space.compiled.problem.numBoolVars).also { arr ->
        for ((name, id) in space.compiled.boolVarIdByName) arr[id] = name
    }
    private val intNameById: Array<String?> = arrayOfNulls<String>(space.compiled.problem.numIntVars).also { arr ->
        for ((name, id) in space.compiled.intVarIdByName) arr[id] = name
    }

    override fun encode(sample: Sample, context: Context): Vector {
        val out = vectors.zeroVector(layout.featureSize)
        for (b in 0 until layout.numBoolDecisions) {
            if (boolDecisionActive(b, sample)) {
                out[layout.boolDecisionsStart + b] = if (sample.bools[b]) 1f else 0f
            }
        }
        for (i in 0 until layout.numIntDecisions) {
            if (intDecisionActive(i, sample)) {
                out[layout.intDecisionsStart + i] = sample.ints[i].toFloat()
            }
        }
        for ((handle, idx) in layout.boolContextIndex) {
            out[idx] = if (context[handle]) 1f else 0f
        }
        for ((handle, idx) in layout.intContextIndex) {
            out[idx] = context[handle].toFloat()
        }
        for ((interaction, idx) in layout.interactionIndex) {
            out[idx] = encodeInteraction(interaction, sample, context).toFloat()
        }
        return out
    }

    /**
     * Build a klause objective whose minimiser is the best-scoring feasible [Sample]
     * under [weights] given [context]. Context-feature contributions are folded into
     * the objective's `constant` (they're invariant during this choose). `maximize=true`
     * flips signs so klause's minimiser searches for the bandit's argmax.
     *
     * Interaction weights fold accordingly:
     *  - `ctx × decision_bool` adds `weight * ctx_value` to the bool's klause coefficient
     *  - `ctx × decision_int`  adds `weight * ctx_value` to the int's klause coefficient
     *  - `ctx × ctx` contributes a constant `weight * ctx1_value * ctx2_value`
     */
    fun toObjective(weights: VectorView, bias: Float, context: Context, maximize: Boolean): LinearObjective {
        require(weights.size == layout.featureSize) {
            "weights size ${weights.size} must match feature layout ${layout.featureSize}"
        }
        val sign = if (maximize) -1.0 else 1.0
        val boolWeights = DoubleArray(layout.numBoolDecisions) { weights[layout.boolDecisionsStart + it].toDouble() }
        val intCoefficients = DoubleArray(layout.numIntDecisions) { weights[layout.intDecisionsStart + it].toDouble() }
        var constant = bias.toDouble()
        for ((handle, idx) in layout.boolContextIndex) {
            if (context[handle]) constant += weights[idx].toDouble()
        }
        for ((handle, idx) in layout.intContextIndex) {
            constant += weights[idx].toDouble() * context[handle]
        }
        for ((interaction, idx) in layout.interactionIndex) {
            foldInteraction(interaction, weights[idx].toDouble(), context, boolWeights, intCoefficients) { c ->
                constant += c
            }
        }
        // Apply minimisation sign last, to the assembled coefficients.
        for (i in boolWeights.indices) boolWeights[i] *= sign
        for (i in intCoefficients.indices) intCoefficients[i] *= sign
        return LinearObjective(boolWeights, intCoefficients, constant = sign * constant)
    }

    private fun boolDecisionActive(id: Int, sample: Sample): Boolean {
        if (id !in optionalBoolIds) return true
        return space.isActive(BoolHandle(boolNameById[id]!!), sample)
    }

    private fun intDecisionActive(id: Int, sample: Sample): Boolean {
        if (id !in optionalIntIds) return true
        val name = intNameById[id]!!
        val domain = space.compiled.problem.intDomains[id]
        return space.isActive(IntHandle(name, domain.min, domain.max), sample)
    }

    private fun encodeInteraction(it: InteractionHandle, sample: Sample, context: Context): Double {
        val l = scalarFor(it.lhs, sample, context)
        val r = scalarFor(it.rhs, sample, context)
        return l * r
    }

    private fun scalarFor(handle: Any, sample: Sample, context: Context): Double = when (handle) {
        is BoolHandle -> {
            val id = space.compiled.boolVarIdByName[handle.name]
                ?: error("interaction references unknown bool decision '${handle.name}'")
            if (!boolDecisionActive(id, sample)) 0.0
            else if (sample.bools[id]) 1.0 else 0.0
        }
        is IntHandle -> {
            val id = space.compiled.intVarIdByName[handle.name]
                ?: error("interaction references unknown int decision '${handle.name}'")
            if (!intDecisionActive(id, sample)) 0.0
            else sample.ints[id].toDouble()
        }
        is BoolContextHandle -> if (context[handle]) 1.0 else 0.0
        is IntContextHandle -> context[handle].toDouble()
        else -> error("unsupported interaction handle: $handle")
    }

    /**
     * Decompose an interaction's contribution to klause's objective. The interaction is
     * `weight * lhs * rhs`. At choose-time context is fixed; if exactly one side is a
     * decision, the other side reduces to a scalar that scales that decision's klause
     * coefficient. If both sides are context, the product is a constant.
     */
    private fun foldInteraction(
        it: InteractionHandle,
        weight: Double,
        context: Context,
        boolWeights: DoubleArray,
        intCoefficients: DoubleArray,
        addToConstant: (Double) -> Unit,
    ) {
        val (decisionSide, contextScalar) = decisionAndContextScalar(it, context)
        if (decisionSide == null) {
            // context × context: pure constant contribution.
            addToConstant(weight * contextScalar)
            return
        }
        when (decisionSide) {
            is BoolHandle -> {
                val id = space.compiled.boolVarIdByName[decisionSide.name]
                    ?: error("interaction references unknown bool decision '${decisionSide.name}'")
                boolWeights[id] += weight * contextScalar
            }
            is IntHandle -> {
                val id = space.compiled.intVarIdByName[decisionSide.name]
                    ?: error("interaction references unknown int decision '${decisionSide.name}'")
                intCoefficients[id] += weight * contextScalar
            }
            else -> error("unsupported decision side in interaction: $decisionSide")
        }
    }

    /** Return (decision-side handle if any, context-scalar contribution). */
    private fun decisionAndContextScalar(it: InteractionHandle, context: Context): Pair<Any?, Double> {
        val lhsIsDecision = it.lhs is BoolHandle || it.lhs is IntHandle
        val rhsIsDecision = it.rhs is BoolHandle || it.rhs is IntHandle
        return when {
            lhsIsDecision && !rhsIsDecision -> it.lhs to ctxScalar(it.rhs, context)
            !lhsIsDecision && rhsIsDecision -> it.rhs to ctxScalar(it.lhs, context)
            !lhsIsDecision && !rhsIsDecision -> null to ctxScalar(it.lhs, context) * ctxScalar(it.rhs, context)
            else -> error("decision × decision interactions are not supported")
        }
    }

    private fun ctxScalar(handle: Any, context: Context): Double = when (handle) {
        is BoolContextHandle -> if (context[handle]) 1.0 else 0.0
        is IntContextHandle -> context[handle].toDouble()
        else -> error("expected context handle, got $handle")
    }
}
