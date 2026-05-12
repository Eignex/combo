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
 * Linear bandit's view of a [CompiledDecisionSpace]: projects a [Sample] into a dense
 * feature vector, and translates a learned weight vector back into a klause
 * [LinearObjective] for choose-time optimisation.
 *
 * Contexts are now ordinary solver variables (pinned per-call via assumptions); the
 * encoder reads them from the sample alongside decisions. The [Context] argument is
 * still accepted on [encode] for parity with the bandit API but is not consulted —
 * the canonical context values are whatever the sample carries.
 *
 * Optional decision variables: when their activation condition evaluates to false on
 * a sample, the corresponding feature slot is zeroed so the linear-model gradient
 * vanishes there.
 *
 * Interactions get extra feature slots. `ctx × decision` interactions fold into the
 * bool/int coefficients of klause's objective; `ctx × ctx` becomes a constant.
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
        for (b in 0 until layout.numBoolVars) {
            if (boolActive(b, sample)) {
                out[layout.boolStart + b] = if (sample.bools[b]) 1f else 0f
            }
        }
        for (i in 0 until layout.numIntVars) {
            if (intActive(i, sample)) {
                out[layout.intStart + i] = sample.ints[i].toFloat()
            }
        }
        for (interaction in layout.interactions) {
            encodeInteractionInto(interaction, sample, out)
        }
        return out
    }

    private fun encodeInteractionInto(it: InteractionHandle, sample: Sample, out: Vector) {
        val start = layout.interactionStart.getValue(it)
        val nominal = it.nominalSide()
        if (nominal == null) {
            val v = scalarFor(it.lhs, sample) * scalarFor(it.rhs, sample)
            out[start] = v.toFloat()
            return
        }
        val ctxSide = if (it.lhs === nominal) it.rhs else it.lhs
        val scalar = scalarFor(ctxSide, sample)
        val indicators = space.compiled.nominalIndicators[nominal.name]
            ?: error("interaction references unknown nominal '${nominal.name}'")
        for ((idx, label) in nominal.labels.withIndex()) {
            val indicatorId = indicators[label]
                ?: error("nominal '${nominal.name}' missing label '$label'")
            if (sample.bools[indicatorId]) out[start + idx] = scalar.toFloat()
        }
    }

    /**
     * Build a klause objective whose minimiser is the best-scoring feasible sample
     * under [weights] given [context]. The caller is responsible for also passing
     * [com.eignex.klause.solver.Assumptions] (built via [assumptionsFor]) to the solver
     * so the context variables are pinned to the values [context] supplies.
     *
     * `maximize = true` flips signs so klause's minimiser searches for the bandit's
     * argmax.
     */
    fun toObjective(weights: VectorView, bias: Float, context: Context, maximize: Boolean): LinearObjective {
        require(weights.size == layout.featureSize) {
            "weights size ${weights.size} must match feature layout ${layout.featureSize}"
        }
        val sign = if (maximize) -1.0 else 1.0
        val boolWeights = DoubleArray(layout.numBoolVars) { weights[layout.boolStart + it].toDouble() }
        val intCoefficients = DoubleArray(layout.numIntVars) { weights[layout.intStart + it].toDouble() }
        var constant = bias.toDouble()
        for (interaction in layout.interactions) {
            foldInteraction(interaction, weights, context, boolWeights, intCoefficients) { c -> constant += c }
        }
        for (i in boolWeights.indices) boolWeights[i] *= sign
        for (i in intCoefficients.indices) intCoefficients[i] *= sign
        return LinearObjective(boolWeights, intCoefficients, constant = sign * constant)
    }

    /**
     * Build the klause [com.eignex.klause.solver.Assumptions] that pin every context
     * variable in [context] to its supplied value for a choose call. For optional
     * contexts: when present, pin both the isKnown gate (true) and the value; when
     * absent, pin just the isKnown gate to false and let klause's schema-level pinning
     * constraint force the value to its default.
     */
    fun assumptionsFor(context: Context): com.eignex.klause.solver.Assumptions {
        val bools = mutableMapOf<Int, Boolean>()
        val ints = mutableMapOf<Int, Int>()
        for (h in space.contextBools) {
            val gate = h.isKnownGate
            if (gate != null) {
                val gateId = space.compiled.boolVarIdByName[gate.name]
                    ?: error("optional context '${h.name}' lost its isKnown gate '${gate.name}'")
                if (context.isPresent(h)) {
                    bools[gateId] = true
                    val valueId = space.compiled.boolVarIdByName[h.name] ?: continue
                    bools[valueId] = context[h]
                } else {
                    bools[gateId] = false
                }
            } else {
                val id = space.compiled.boolVarIdByName[h.name] ?: continue
                bools[id] = context[h]
            }
        }
        for (h in space.contextInts) {
            val gate = h.isKnownGate
            if (gate != null) {
                val gateId = space.compiled.boolVarIdByName[gate.name]
                    ?: error("optional context '${h.name}' lost its isKnown gate '${gate.name}'")
                if (context.isPresent(h)) {
                    bools[gateId] = true
                    val valueId = space.compiled.intVarIdByName[h.name] ?: continue
                    ints[valueId] = context[h]
                } else {
                    bools[gateId] = false
                }
            } else {
                val id = space.compiled.intVarIdByName[h.name] ?: continue
                ints[id] = context[h]
            }
        }
        return com.eignex.klause.solver.Assumptions(bools, ints)
    }

    private fun boolActive(id: Int, sample: Sample): Boolean {
        if (id !in optionalBoolIds) return true
        return space.isActive(BoolHandle(boolNameById[id]!!), sample)
    }

    private fun intActive(id: Int, sample: Sample): Boolean {
        if (id !in optionalIntIds) return true
        val name = intNameById[id]!!
        val d = space.compiled.problem.intDomains[id]
        return space.isActive(IntHandle(name, d.min, d.max), sample)
    }

    private fun scalarFor(handle: Any, sample: Sample): Double = when (handle) {
        is BoolHandle -> {
            val id = space.compiled.boolVarIdByName[handle.name]
                ?: error("interaction references unknown bool '${handle.name}'")
            if (!boolActive(id, sample)) 0.0 else if (sample.bools[id]) 1.0 else 0.0
        }
        is IntHandle -> {
            val id = space.compiled.intVarIdByName[handle.name]
                ?: error("interaction references unknown int '${handle.name}'")
            if (!intActive(id, sample)) 0.0 else sample.ints[id].toDouble()
        }
        is NominalHandle -> error("nominal handle should not be a direct scalar; expand via interaction logic")
        is BoolContextHandle -> scalarFor(handle.klauseHandle, sample)
        is IntContextHandle -> scalarFor(handle.klauseHandle, sample)
        else -> error("unsupported handle: $handle")
    }

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
        val lhsIsDecision = isDecisionScalar(it.lhs)
        val rhsIsDecision = isDecisionScalar(it.rhs)
        when {
            !lhsIsDecision && !rhsIsDecision -> {
                addToConstant(w * ctxScalar(it.lhs, context) * ctxScalar(it.rhs, context))
            }
            lhsIsDecision -> applyDecisionWeight(it.lhs, w * ctxScalar(it.rhs, context), boolWeights, intCoefficients)
            else -> applyDecisionWeight(it.rhs, w * ctxScalar(it.lhs, context), boolWeights, intCoefficients)
        }
    }

    private fun isDecisionScalar(handle: Any): Boolean = when (handle) {
        is BoolContextHandle, is IntContextHandle -> false
        is BoolHandle, is IntHandle, is NominalHandle -> true
        else -> false
    }

    private fun ctxScalar(handle: Any, context: Context): Double = when (handle) {
        is BoolContextHandle -> if (context[handle]) 1.0 else 0.0
        is IntContextHandle -> context[handle].toDouble()
        else -> error("expected context handle, got $handle")
    }

    private fun applyDecisionWeight(
        decision: Any,
        weight: Double,
        boolWeights: DoubleArray,
        intCoefficients: DoubleArray,
    ) {
        when (decision) {
            is BoolHandle -> {
                val id = space.compiled.boolVarIdByName[decision.name]
                    ?: error("interaction references unknown bool '${decision.name}'")
                boolWeights[id] += weight
            }
            is IntHandle -> {
                val id = space.compiled.intVarIdByName[decision.name]
                    ?: error("interaction references unknown int '${decision.name}'")
                intCoefficients[id] += weight
            }
            else -> error("unsupported decision in interaction: $decision")
        }
    }
}
