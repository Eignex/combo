package combo.bandit.glm

import com.eignex.klause.ast.BoolExpr
import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.IntHandle
import com.eignex.klause.schema.NominalHandle
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
        for (interaction in layout.interactions) {
            encodeInteractionInto(interaction, sample, context, out)
        }
        return out
    }

    private fun encodeInteractionInto(it: InteractionHandle, sample: Sample, context: Context, out: Vector) {
        val start = layout.interactionStart.getValue(it)
        val nominal = it.nominalSide()
        if (nominal == null) {
            val v = scalarFor(it.lhs, sample, context) * scalarFor(it.rhs, sample, context)
            out[start] = v.toFloat()
            return
        }
        // Nominal × context: K slots, one per label. The active label's slot carries
        // ctx_scalar; the rest stay zero.
        val ctxSide = if (it.lhs === nominal) it.rhs else it.lhs
        val ctxScalar = ctxScalar(ctxSide, context)
        val indicators = space.compiled.nominalIndicators[nominal.name]
            ?: error("interaction references unknown nominal '${nominal.name}'")
        for ((idx, label) in nominal.labels.withIndex()) {
            val indicatorId = indicators[label] ?: error("nominal '${nominal.name}' missing label '$label'")
            if (sample.bools[indicatorId]) {
                out[start + idx] = ctxScalar.toFloat()
            }
        }
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
        for (interaction in layout.interactions) {
            foldInteraction(interaction, weights, context, boolWeights, intCoefficients) { c ->
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
     * Decompose an interaction's contribution(s) to klause's objective. For scalar
     * decision interactions, one slot folds into the decision's klause coefficient.
     * For nominal interactions, each label's per-slot weight folds into the matching
     * indicator's klause coefficient. context × context contributes a constant.
     */
    private fun foldInteraction(
        it: InteractionHandle,
        allWeights: VectorView,
        context: Context,
        boolWeights: DoubleArray,
        intCoefficients: DoubleArray,
        addToConstant: (Double) -> Unit,
    ) {
        val start = layout.interactionStart.getValue(it)
        val nominal = it.nominalSide()
        if (nominal != null) {
            val ctxSide = if (it.lhs === nominal) it.rhs else it.lhs
            val ctxScalar = ctxScalar(ctxSide, context)
            val indicators = space.compiled.nominalIndicators[nominal.name]
                ?: error("interaction references unknown nominal '${nominal.name}'")
            for ((idx, label) in nominal.labels.withIndex()) {
                val w = allWeights[start + idx].toDouble()
                val indicatorId = indicators[label]
                    ?: error("nominal '${nominal.name}' missing label '$label'")
                boolWeights[indicatorId] += w * ctxScalar
            }
            return
        }
        val w = allWeights[start].toDouble()
        val lhsIsDecision = it.lhs is BoolHandle || it.lhs is IntHandle
        val rhsIsDecision = it.rhs is BoolHandle || it.rhs is IntHandle
        when {
            !lhsIsDecision && !rhsIsDecision -> {
                addToConstant(w * ctxScalar(it.lhs, context) * ctxScalar(it.rhs, context))
            }
            lhsIsDecision -> applyDecisionScaledWeight(it.lhs, w * ctxScalar(it.rhs, context), boolWeights, intCoefficients)
            else -> applyDecisionScaledWeight(it.rhs, w * ctxScalar(it.lhs, context), boolWeights, intCoefficients)
        }
    }

    private fun applyDecisionScaledWeight(
        decision: Any,
        weight: Double,
        boolWeights: DoubleArray,
        intCoefficients: DoubleArray,
    ) {
        when (decision) {
            is BoolHandle -> {
                val id = space.compiled.boolVarIdByName[decision.name]
                    ?: error("interaction references unknown bool decision '${decision.name}'")
                boolWeights[id] += weight
            }
            is IntHandle -> {
                val id = space.compiled.intVarIdByName[decision.name]
                    ?: error("interaction references unknown int decision '${decision.name}'")
                intCoefficients[id] += weight
            }
            else -> error("unsupported decision side in interaction: $decision")
        }
    }

    private fun ctxScalar(handle: Any, context: Context): Double = when (handle) {
        is BoolContextHandle -> if (context[handle]) 1.0 else 0.0
        is IntContextHandle -> context[handle].toDouble()
        else -> error("expected context handle, got $handle")
    }
}
