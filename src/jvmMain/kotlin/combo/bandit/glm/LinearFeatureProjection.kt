package combo.bandit.glm

import com.eignex.klause.ast.BoolExpr
import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.IntHandle
import com.eignex.klause.schema.NominalHandle
import com.eignex.klause.solver.LinearObjective
import com.eignex.klause.solver.Sample
import combo.decisions.BanditSample
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

    override fun encode(sample: BanditSample, context: Context): Vector {
        val out = vectors.zeroVector(layout.featureSize)
        for (b in 0 until layout.numBoolVars) {
            if (boolActive(b, sample.sample)) {
                out[layout.boolStart + b] = if (sample.bools[b]) 1f else 0f
            }
        }
        for (i in 0 until layout.numIntVars) {
            if (intActive(i, sample.sample)) {
                // For float slots we prefer the dithered continuous value (carried by
                // BanditSample) over the bucket midpoint — same dimensional units,
                // sub-bucket precision. realValue handles the identity case for true ints.
                val scaling = layout.floatScaling[i]
                val v = if (scaling != null) floatFeatureFor(i, sample)
                        else layout.realValue(i, sample.ints[i])
                out[layout.intStart + i] = v.toFloat()
            }
        }
        for (interaction in layout.interactions) {
            encodeInteractionInto(interaction, sample, out)
        }
        return out
    }

    /** Resolve the continuous float value at int slot [intVarId] from the bandit
     *  sample's dither, falling back to the bucket midpoint when no dither is present. */
    private fun floatFeatureFor(intVarId: Int, sample: BanditSample): Double {
        val name = intNameById[intVarId] ?: return sample.sample.ints[intVarId].toDouble()
        val spec = space.compiled.floatDecoders[name]
            ?: return sample.sample.ints[intVarId].toDouble()
        val handle = com.eignex.klause.schema.FloatHandle(name, spec.min, spec.max, spec.buckets)
        return sample.float(handle, space)
    }

    private fun encodeInteractionInto(it: InteractionHandle, sample: BanditSample, out: Vector) {
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

    private fun scalarFor(handle: Any, sample: BanditSample): Double = when (handle) {
        is BoolHandle -> {
            val id = space.compiled.boolVarIdByName[handle.name]
                ?: error("interaction references unknown bool '${handle.name}'")
            if (!boolActive(id, sample.sample)) 0.0 else if (sample.bools[id]) 1.0 else 0.0
        }
        is IntHandle -> {
            val id = space.compiled.intVarIdByName[handle.name]
                ?: error("interaction references unknown int '${handle.name}'")
            if (!intActive(id, sample.sample)) 0.0
            else if (id in layout.floatScaling) floatFeatureFor(id, sample)
            else layout.realValue(id, sample.ints[id])
        }
        is NominalHandle -> error("nominal handle should not be a direct scalar; expand via interaction logic")
        is BoolContextHandle -> scalarFor(handle.klauseHandle, sample)
        is IntContextHandle -> scalarFor(handle.klauseHandle, sample)
        else -> error("unsupported handle: $handle")
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
        // Per-int weights enter klause as bucket-coefficients. For bucketed-float slots
        // that means scaling the model weight by the bucket→real scale; the leftover
        // offset (weight · min) rolls into the objective's constant so the round-trip
        // model_weight · real_value = bucket_coefficient · bucket + constant_contribution
        // is exact.
        val intCoefficients = DoubleArray(layout.numIntVars)
        var constant = bias.toDouble()
        for (i in 0 until layout.numIntVars) {
            val w = weights[layout.intStart + i].toDouble()
            val (bucketCoeff, constantContribution) = layout.coefficientForInt(i, w)
            intCoefficients[i] = bucketCoeff
            constant += constantContribution
        }
        for (interaction in layout.interactions) {
            foldInteraction(interaction, weights, context, boolWeights, intCoefficients) { c -> constant += c }
        }
        for (i in boolWeights.indices) boolWeights[i] *= sign
        for (i in intCoefficients.indices) intCoefficients[i] *= sign
        return LinearObjective(boolWeights, intCoefficients, constant = sign * constant)
    }

    /** Convenience: delegates to [CompiledDecisionSpace.assumptionsFor]. */
    fun assumptionsFor(context: Context) = space.assumptionsFor(context)

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
            lhsIsDecision -> {
                val c = applyDecisionWeight(it.lhs, w * ctxScalar(it.rhs, context), boolWeights, intCoefficients)
                if (c != 0.0) addToConstant(c)
            }
            else -> {
                val c = applyDecisionWeight(it.rhs, w * ctxScalar(it.lhs, context), boolWeights, intCoefficients)
                if (c != 0.0) addToConstant(c)
            }
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
    ): Double {
        // Returns the constant-term contribution introduced by float scaling, if any.
        when (decision) {
            is BoolHandle -> {
                val id = space.compiled.boolVarIdByName[decision.name]
                    ?: error("interaction references unknown bool '${decision.name}'")
                boolWeights[id] += weight
                return 0.0
            }
            is IntHandle -> {
                val id = space.compiled.intVarIdByName[decision.name]
                    ?: error("interaction references unknown int '${decision.name}'")
                val (bucketCoeff, constantContribution) = layout.coefficientForInt(id, weight)
                intCoefficients[id] += bucketCoeff
                return constantContribution
            }
            else -> error("unsupported decision in interaction: $decision")
        }
    }
}
