package com.eignex.combo.decisions

import com.eignex.klause.compile.CompiledSchema
import com.eignex.klause.model.BoolExpr
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.IntHandle
import com.eignex.klause.schema.NominalHandle
import com.eignex.klause.solver.Sample
import com.eignex.klause.arithmetic.real.exactRealValue
import com.eignex.klause.arithmetic.real.compareExactly
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.util.bigIntOf
import com.eignex.klause.util.compareTo

/**
 * Structural snapshot of a compiled [DecisionSpace]: a klause [CompiledSchema] paired
 * with the unified [DecisionSpaceDef] that's the single source of truth for the schema.
 *
 * Bandit-family-specific projections (linear feature vectors, tree split tables, …)
 * live in their own modules and consume this as input. The activation map lets each
 * projection decide how to handle inactive rows — linear bandits zero the slot, tree
 * bandits can emit an `isPresent` indicator, etc.
 *
 * This type is intentionally bandit-agnostic. It carries no feature layout, no
 * `encode`/`toObjective` — those concerns belong with the bandit that needs them.
 */
class CompiledDecisionSpace internal constructor(
    compiled: CompiledSchema?,
    val contextBools: List<BoolContextHandle>,
    val contextInts: List<IntContextHandle>,
    val interactions: List<InteractionHandle>,
    /** Single serialisable record of the entire feature model. */
    val definition: DecisionSpaceDef,
    /** For each conditionally-present klause variable, the boolean expression that
     *  must hold for the variable to be active. */
    val activeConditions: Map<String, BoolExpr>,
    /** For each sub-space mounted via `optionalDecisionSpace`, the auto-allocated gate
     *  variable that controls its activation. */
    val gates: Map<SubSpace, BoolHandle>,
    /** Required contexts in real units, including explicit grids. */
    val contextFloats: List<FloatContextHandle> = emptyList(),
    nativeCoordinates: DecisionCoordinates? = null,
) {
    /** Shared native coordinates, independent of whether a schema was used to construct the model. */
    val coordinates: DecisionCoordinates = nativeCoordinates ?: DecisionCoordinates.of(requireNotNull(compiled))

    /** Existing schema API for callers using the Kotlin/schema DSL. Raw problem consumers use [coordinates]. */
    val compiled: CompiledSchema get() = requireNotNull(coordinates.schema) { "this model is a native Problem" }

    /** The complete original Klause problem, without rebuilding its domains or factors. */
    val problem: com.eignex.klause.ir.Problem get() = coordinates.problem

    internal val numericDomains: Map<String, NumericDomain> = if (coordinates.schema != null) {
        definition.numericDomains()
    } else {
        nativeNumericDomains(problem)
    }

    /** Native problem entry point shared by all feature projections and optimization policies. */
    companion object {
        /** Retain all native columns, factors, domain declarations, and metadata unchanged. */
        fun fromProblem(problem: com.eignex.klause.ir.Problem): CompiledDecisionSpace = CompiledDecisionSpace(
            null, emptyList(), emptyList(), emptyList(), DecisionSpaceDef(), emptyMap(), emptyMap(),
            nativeCoordinates = DecisionCoordinates.of(problem),
        )
    }

    /** Convenience: the auto-allocated gate for an optional sub-space. Null when the
     *  sub-space was mounted via plain `decisionSpace { … }` (unconditional). */
    fun gateOf(sub: SubSpace): BoolHandle? = gates[sub]

    /** Decode a context bool's value from a sample. */
    fun decode(handle: BoolContextHandle, sample: Sample): Boolean = coordinates.decode(handle.klauseHandle, sample)

    /** Decode a context int's value from a sample. */
    fun decode(handle: IntContextHandle, sample: Sample): Int =
        Math.toIntExact(coordinates.decode(handle.klauseHandle, sample))

    fun isActive(handle: BoolContextHandle, sample: Sample): Boolean = isActive(handle.klauseHandle, sample)
    fun isActive(handle: IntContextHandle, sample: Sample): Boolean = isActive(handle.klauseHandle, sample)

    fun isOptional(handle: BoolContextHandle): Boolean = isOptional(handle.klauseHandle)
    fun isOptional(handle: IntContextHandle): Boolean = isOptional(handle.klauseHandle)

    /**
     * Build the klause [Assumptions] that pin every context variable in [context] to
     * its supplied value for a choose call. For optional contexts: when present, pin
     * both the isKnown gate (true) and the value; when absent, pin just the isKnown
     * gate to false and let the schema's pinning constraint force the value to its
     * default.
     */
    fun assumptionsFor(context: Context): Assumptions {
        require(!context.hasRealValues) { "real context requires conditionedProblem and discreteAssumptionsFor" }
        return discreteAssumptionsFor(context)
    }

    /** Discrete portion of a context; combine with [conditionedProblem] for its real pins. */
    fun discreteAssumptionsFor(context: Context): Assumptions {
        val bools = mutableMapOf<Int, Boolean>()
        val ints = mutableMapOf<Int, Long>()
        for (h in contextBools) {
            val gate = h.isKnownGate
            if (gate != null) {
                val gateId = coordinates.boolVarIdByName[gate.name]
                    ?: error("optional context '${h.name}' lost its isKnown gate '${gate.name}'")
                if (context.isPresent(h)) {
                    bools[gateId] = true
                    val valueId = coordinates.boolVarIdByName[h.name] ?: continue
                    bools[valueId] = context[h]
                } else {
                    bools[gateId] = false
                }
            } else {
                val id = coordinates.boolVarIdByName[h.name] ?: continue
                bools[id] = context[h]
            }
        }
        for (h in contextInts) {
            val gate = h.isKnownGate
            if (gate != null) {
                val gateId = coordinates.boolVarIdByName[gate.name]
                    ?: error("optional context '${h.name}' lost its isKnown gate '${gate.name}'")
                if (context.isPresent(h)) {
                    bools[gateId] = true
                    val valueId = coordinates.intVarIdByName[h.name] ?: continue
                    ints[valueId] = context[h].toLong()
                } else {
                    bools[gateId] = false
                }
            } else {
                val id = coordinates.intVarIdByName[h.name] ?: continue
                ints[id] = context[h].toLong()
            }
        }
        for (handle in contextFloats) {
            val value = context[handle]
            if (coordinates.minimize(handle.klauseHandle).realCoefficients.any { it != 0.0 }) continue
            coordinates.intVarIdByName[handle.name]?.let { id ->
                ints[id] = floatBucket(coordinates.floatDecoders.getValue(handle.name), value)
            }
        }
        context.nativeAssumptions.forEachBool { id, _ ->
            require(id in 0 until problem.numBoolVars) { "unknown Boolean $id" }
        }
        context.nativeAssumptions.forEachInt { id, value ->
            require(id in 0 until problem.numIntVars) { "unknown integer $id" }
            val bounds = problem.intBounds
            require((!bounds.hasLower(id) || value >= bounds.lower(id)) &&
                (!bounds.hasUpper(id) || value <= bounds.upper(id))) { "integer $id outside its bounds" }
            require(problem.intDomainOrNull(id)?.let { value in it } != false) { "integer $id outside its value set" }
        }
        return Assumptions(bools, ints).mergedWith(context.nativeAssumptions)
    }

    /** Verify both real and discrete context pins against a native assignment. */
    fun matches(sample: Sample, context: Context): Boolean =
        matches(sample, discreteAssumptionsFor(context)) && contextFloats.all {
            coordinates.decode(it.klauseHandle, sample) == context[it]
        } && context.exactInts.all { (id, value) ->
            id in 0 until problem.numIntVars && sample.exactIntValue(id) == value
        } && context.exactReals.all { (id, value) ->
            id in 0 until problem.numRealVars && (
                sample.rationalReal(id)
                ) == value
        } && context.nativeReals.all { (id, value) ->
            id in 0 until problem.numRealVars && sample.rationalReal(id) == BigFraction.ofDouble(value)
        } && context.exactRealValues.all { (id, value) ->
            id in 0 until problem.numRealVars && sample.exactRealValue(id).compareExactly(value) == 0
        }

    /** True iff [sample] satisfies every pin in [assumptions]. */
    fun matches(sample: Sample, assumptions: Assumptions): Boolean {
        for ((id, v) in assumptions.bools) if (sample.boolValue(id) != v) return false
        for ((id, v) in assumptions.ints) {
            if (sample.exactIntValue(id) != bigIntOf(v)) return false
        }
        val deductions = assumptions.deductions
        deductions.forEachIntMin { id, value -> if (sample.exactIntValue(id) < bigIntOf(value)) return false }
        deductions.forEachIntMax { id, value -> if (sample.exactIntValue(id) > bigIntOf(value)) return false }
        deductions.forEachIntHole { id, value -> if (sample.exactIntValue(id) == bigIntOf(value)) return false }
        deductions.forEachIntSet { id, values ->
            if (values.none { sample.exactIntValue(id) == bigIntOf(it) }) return false
        }
        return true
    }

    /** True if [handle] is conditionally present (i.e. has an activation condition). */
    fun isOptional(handle: BoolHandle): Boolean = handle.name in activeConditions
    fun isOptional(handle: IntHandle): Boolean = handle.name in activeConditions
    fun isOptional(handle: NominalHandle): Boolean = handle.name in activeConditions

    /** Evaluate `handle`'s activation against a klause [Sample]. Always-active variables
     *  return true. Variables with conditional presence return whether their gate(s)
     *  are satisfied in this sample. */
    fun isActive(handle: BoolHandle, sample: Sample): Boolean =
        activeConditions[handle.name]?.let { evaluate(it, sample) } ?: true

    fun isActive(handle: IntHandle, sample: Sample): Boolean =
        activeConditions[handle.name]?.let { evaluate(it, sample) } ?: true

    fun isActive(handle: NominalHandle, sample: Sample): Boolean =
        activeConditions[handle.name]?.let { evaluate(it, sample) } ?: true

    /** Evaluate an arbitrary klause [BoolExpr] against [sample]. Same machinery the
     *  active-condition checks use; exposed so tree-side split predicates can share
     *  one evaluator with user-facing constraints. */
    fun evaluate(expr: BoolExpr, sample: Sample): Boolean = evaluateBool(expr, sample, coordinates)
}

internal fun InteractionHandle.toDef(): InteractionDef =
    InteractionDef(name = name, lhs = lhs.toScalarRef(), rhs = rhs.toScalarRef())

private fun Any.toScalarRef(): ScalarRef = when (this) {
    is BoolHandle -> ScalarRef(ScalarKind.BoolDecision, name.split("."))
    is IntHandle -> ScalarRef(ScalarKind.IntDecision, name.split("."))
    is com.eignex.klause.schema.FloatHandle -> ScalarRef(ScalarKind.FloatDecision, name.split("."))
    is FloatContextHandle -> ScalarRef(ScalarKind.FloatContext, listOf("context", name))
    is NominalHandle -> ScalarRef(ScalarKind.NominalDecision, name.split("."), labels)
    is BoolContextHandle -> ScalarRef(ScalarKind.BoolContext, listOf("context", name))
    is IntContextHandle -> ScalarRef(ScalarKind.IntContext, listOf("context", name))
    else -> error("unsupported handle in interaction: $this")
}

private fun evaluateBool(expr: BoolExpr, sample: Sample, compiled: DecisionCoordinates): Boolean = when (expr) {
    is com.eignex.klause.model.BoolRef -> {
        val id = compiled.boolVarIdByName[expr.name]
            ?: error("evaluating active condition: unknown bool variable '${expr.name}'")
        val raw = sample.boolValue(id)
        if (expr.negated) !raw else raw
    }

    is com.eignex.klause.model.Not -> !evaluateBool(expr.child, sample, compiled)

    is com.eignex.klause.model.And -> expr.children.all { evaluateBool(it, sample, compiled) }

    is com.eignex.klause.model.Or -> expr.children.any { evaluateBool(it, sample, compiled) }

    is com.eignex.klause.model.Implies -> !evaluateBool(
        expr.left,
        sample,
        compiled,
    ) || evaluateBool(expr.right, sample, compiled)

    is com.eignex.klause.model.Iff -> evaluateBool(
        expr.left,
        sample,
        compiled,
    ) == evaluateBool(expr.right, sample, compiled)

    is com.eignex.klause.model.NominalEq -> {
        val indicators = compiled.nominalIndicators[expr.name]
            ?: error("evaluating active condition: unknown nominal '${expr.name}'")
        val bitId = indicators[expr.label]
            ?: error("evaluating active condition: label '${expr.label}' not on nominal '${expr.name}'")
        sample.boolValue(bitId)
    }

    else -> error("evaluating active condition: unsupported expression $expr")
}


private fun Sample.rationalReal(id: Int): BigFraction? {
    symbolicReals?.get(id)?.let { return it.rational }
    algebraicReals?.get(id)?.let { return it.rational }
    return exactReals?.get(id) ?: BigFraction.ofDouble(approximateRealValue(id))
}
