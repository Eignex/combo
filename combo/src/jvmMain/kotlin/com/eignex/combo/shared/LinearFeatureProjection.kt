package com.eignex.combo.shared

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.BoolContextHandle
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.FeatureEncoder
import com.eignex.combo.decisions.IntContextHandle
import com.eignex.combo.decisions.InteractionHandle
import com.eignex.combo.decisions.Scale
import com.eignex.combo.shared.integerObjective
import com.eignex.klause.model.BoolExpr
import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.IntHandle
import com.eignex.klause.schema.NominalHandle
import com.eignex.klause.solver.Sample
import com.eignex.klause.solver.objective.LinearObjective
import com.eignex.koblas.DenseVector
import com.eignex.koblas.Vector
import com.eignex.koblas.scale

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
 * bool/int coefficients of klause's objective; `ctx × ctx` is constant during optimization.
 * Continuous floats occupy real-valued feature slots and objective coefficients.
 * Nonlinear scale hints transform numeric features and interactions. [toObjective]
 * rejects such projections; their scores require feasible candidate search.
 */
class LinearFeatureProjection(override val space: CompiledDecisionSpace) : FeatureEncoder<Vector> {

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
        val out = DoubleArray(layout.featureSize)
        for (b in 0 until layout.numBoolVars) {
            if (boolActive(b, sample.sample)) {
                out[layout.boolStart + b] = if (sample.sample.boolValue(b)) 1.0 else 0.0
            }
        }
        for (i in 0 until layout.numIntVars) {
            if (intActive(i, sample.sample)) {
                // Decode any finite representation selected by Klause in the variable's
                // declared units; Combo does not choose a representation of its own.
                val scaling = layout.floatSpecs[i]
                val v = if (scaling != null) {
                    floatFeatureFor(i, sample)
                } else {
                    layout.realValue(i, sample.sample.intValue(i))
                }
                out[layout.intStart + i] = feature(intNameById[i], v)
            }
        }
        for ((id, handle) in layout.realHandles) {
            val active = space.activeConditions[handle.name]?.let { space.evaluate(it, sample.sample) } ?: true
            if (active) out[layout.realStart + id] = feature(handle.name, sample.float(handle, space))
        }
        for (interaction in layout.interactions) {
            encodeInteractionInto(interaction, sample, out)
        }
        return DenseVector.of(out)
    }

    /** Whether feature scaling prevents conversion into a linear objective in source coordinates. */
    val hasNonlinearScales: Boolean = space.numericDomains.values.any { it.scale != Scale.Linear }

    private fun feature(name: String?, value: Double): Double = space.numericDomains[name]?.feature(value) ?: value

    /** Decode a float stored in an integer coordinate by Klause. */
    private fun floatFeatureFor(intVarId: Int, sample: BanditSample): Double {
        val name = intNameById[intVarId] ?: return sample.sample.intValue(intVarId).toDouble()
        val spec = space.compiled.floatDecoders[name]
            ?: return sample.sample.intValue(intVarId).toDouble()
        val handle = com.eignex.klause.schema.FloatHandle(name, spec.min, spec.max)
        return sample.float(handle, space)
    }

    private fun encodeInteractionInto(it: InteractionHandle, sample: BanditSample, out: DoubleArray) {
        val start = layout.interactionStart.getValue(it)
        val nominal = it.nominalSide()
        if (nominal == null) {
            out[start] = scalarFor(it.lhs, sample) * scalarFor(it.rhs, sample)
            return
        }
        val ctxSide = if (it.lhs === nominal) it.rhs else it.lhs
        val scalar = scalarFor(ctxSide, sample)
        val indicators = space.compiled.nominalIndicators[nominal.name]
            ?: error("interaction references unknown nominal '${nominal.name}'")
        for ((idx, label) in nominal.labels.withIndex()) {
            val indicatorId = indicators[label]
                ?: error("nominal '${nominal.name}' missing label '$label'")
            if (sample.sample.boolValue(indicatorId)) out[start + idx] = scalar
        }
    }

    private fun scalarFor(handle: Any, sample: BanditSample): Double = when (handle) {
        is BoolHandle -> {
            val id = space.compiled.boolVarIdByName[handle.name]
                ?: error("interaction references unknown bool '${handle.name}'")
            if (!boolActive(id, sample.sample)) {
                0.0
            } else if (sample.sample.boolValue(id)) {
                1.0
            } else {
                0.0
            }
        }

        is IntHandle -> {
            val id = space.compiled.intVarIdByName[handle.name]
                ?: error("interaction references unknown int '${handle.name}'")
            if (!intActive(id, sample.sample)) {
                0.0
            } else if (id in layout.floatSpecs) {
                feature(handle.name, floatFeatureFor(id, sample))
            } else {
                feature(handle.name, layout.realValue(id, sample.sample.intValue(id)))
            }
        }

        is NominalHandle -> error("nominal handle should not be a direct scalar; expand via interaction logic")

        is BoolContextHandle -> scalarFor(handle.klauseHandle, sample)

        is IntContextHandle -> scalarFor(handle.klauseHandle, sample)

        else -> error("unsupported handle: $handle")
    }

    /**
     * Build a klause objective whose minimiser is the best-scoring feasible sample
     * under [weights] given [context]. The caller is responsible for also passing
     * [com.eignex.klause.propagation.Assumptions] (built via [assumptionsFor]) to the solver
     * so the context variables are pinned to the values [context] supplies.
     *
     * `maximize = true` flips signs so klause's minimiser searches for the bandit's
     * argmax. Discrete coefficients share a positive scale before rounding to Klause's
     * integer objective (about nine decimal digits). [bias] and other constant offsets
     * are omitted because they do not affect the winning sample.
     */
    fun toObjective(
        weights: Vector,
        @Suppress(
            "UNUSED_PARAMETER",
        ) bias: Double,
        context: Context,
        maximize: Boolean,
    ): LinearObjective {
        require(!hasNonlinearScales) { "nonlinear scales require candidate scoring, not a linear source objective" }
        require(weights.size == layout.featureSize) {
            "weights size ${weights.size} must match feature layout ${layout.featureSize}"
        }
        val sign = if (maximize) -1.0 else 1.0
        val boolWeights = DoubleArray(layout.numBoolVars) { weights[layout.boolStart + it] }
        // Per-int weights enter klause as bucket-coefficients. For bucketed-float slots
        // that means scaling by the bucket→real step. Constant offsets do not affect
        // the optimum and are omitted, as is the model bias.
        val intCoefficients = DoubleArray(layout.numIntVars)
        for (i in 0 until layout.numIntVars) {
            val w = weights[layout.intStart + i]
            val (bucketCoeff, _) = layout.coefficientForInt(i, w)
            intCoefficients[i] = bucketCoeff
        }
        for (interaction in layout.interactions) {
            foldInteraction(interaction, weights, context, boolWeights, intCoefficients)
        }
        DenseVector.wrap(boolWeights).scale(sign)
        DenseVector.wrap(intCoefficients).scale(sign)
        val realCoefficients = DoubleArray(layout.numRealVars) { sign * weights[layout.realStart + it] }
        return integerObjective(space.compiled.problem, boolWeights, intCoefficients, realCoefficients)
    }

    /** Convenience: delegates to [CompiledDecisionSpace.assumptionsFor]. */
    fun assumptionsFor(context: Context) = space.assumptionsFor(context)

    private fun boolActive(id: Int, sample: Sample): Boolean {
        if (id !in optionalBoolIds) return true
        return space.isActive(BoolHandle(boolNameById[id]!!), sample)
    }

    private fun intActive(id: Int, sample: Sample): Boolean {
        if (id !in optionalIntIds) return true
        return space.evaluate(optionalIntIds.getValue(id), sample)
    }

    private fun foldInteraction(
        it: InteractionHandle,
        allWeights: Vector,
        context: Context,
        boolWeights: DoubleArray,
        intCoefficients: DoubleArray,
    ) {
        val start = layout.interactionStart.getValue(it)
        val nominal = it.nominalSide()
        if (nominal != null) {
            val ctxSide = if (it.lhs === nominal) it.rhs else it.lhs
            val ctxScalar = ctxScalar(ctxSide, context)
            val indicators = space.compiled.nominalIndicators[nominal.name]
                ?: error("interaction references unknown nominal '${nominal.name}'")
            for ((idx, label) in nominal.labels.withIndex()) {
                val w = allWeights[start + idx]
                val indicatorId = indicators[label]
                    ?: error("nominal '${nominal.name}' missing label '$label'")
                boolWeights[indicatorId] += w * ctxScalar
            }
            return
        }
        val w = allWeights[start]
        val lhsIsDecision = isDecisionScalar(it.lhs)
        val rhsIsDecision = isDecisionScalar(it.rhs)
        when {
            !lhsIsDecision && !rhsIsDecision -> {
                // Context-only interactions are constant during optimization.
            }

            lhsIsDecision -> {
                applyDecisionWeight(it.lhs, w * ctxScalar(it.rhs, context), boolWeights, intCoefficients)
            }

            else -> {
                applyDecisionWeight(it.rhs, w * ctxScalar(it.lhs, context), boolWeights, intCoefficients)
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
                val (bucketCoeff, _) = layout.coefficientForInt(id, weight)
                intCoefficients[id] += bucketCoeff
            }

            else -> error("unsupported decision in interaction: $decision")
        }
    }
}
