package combo.decisions

import com.eignex.klause.ast.BoolExpr
import com.eignex.klause.ast.SchemaEntry
import com.eignex.klause.compile.CompiledProblem
import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.IntHandle
import com.eignex.klause.schema.NominalHandle
import com.eignex.klause.solver.Sample
import com.eignex.skema.SchemaDef

/**
 * Structural snapshot of a compiled [DecisionSpace]: a klause [CompiledProblem] for the
 * decision side plus the lists of context handles declared on the space, and the per-
 * variable activation conditions extracted from any optional sub-models.
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
    val compiled: CompiledProblem,
    val contextBools: List<BoolContextHandle>,
    val contextInts: List<IntContextHandle>,
    val schemaDef: SchemaDef<SchemaEntry>,
    /** For each conditionally-present klause variable, the boolean expression that
     *  must hold for the variable to be active. Absent for always-active variables. */
    val activeConditions: Map<String, BoolExpr>,
    /** For each sub-model mounted via `optionalSubmodel`, the auto-allocated gate
     *  variable that controls its activation. */
    val gates: Map<SubSpace, BoolHandle>,
) {
    /** Convenience: the auto-allocated gate for an optional sub-model. Null when the
     *  sub-model was mounted via plain `submodel { … }` (unconditional). */
    fun gateOf(sub: SubSpace): BoolHandle? = gates[sub]

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

    private fun evaluate(expr: BoolExpr, sample: Sample): Boolean = evaluateBool(expr, sample, compiled)
}

private fun evaluateBool(
    expr: BoolExpr,
    sample: Sample,
    compiled: CompiledProblem,
): Boolean = when (expr) {
    is com.eignex.klause.ast.BoolRef -> {
        val id = compiled.boolVarIdByName[expr.name]
            ?: error("evaluating active condition: unknown bool variable '${expr.name}'")
        val raw = sample.bools[id]
        if (expr.negated) !raw else raw
    }
    is com.eignex.klause.ast.Not -> !evaluateBool(expr.child, sample, compiled)
    is com.eignex.klause.ast.And -> expr.children.all { evaluateBool(it, sample, compiled) }
    is com.eignex.klause.ast.Or -> expr.children.any { evaluateBool(it, sample, compiled) }
    is com.eignex.klause.ast.Implies -> !evaluateBool(expr.left, sample, compiled) || evaluateBool(expr.right, sample, compiled)
    is com.eignex.klause.ast.Iff -> evaluateBool(expr.left, sample, compiled) == evaluateBool(expr.right, sample, compiled)
    is com.eignex.klause.ast.NominalEq -> {
        val indicators = compiled.nominalIndicators[expr.name]
            ?: error("evaluating active condition: unknown nominal '${expr.name}'")
        val bitId = indicators[expr.label]
            ?: error("evaluating active condition: label '${expr.label}' not on nominal '${expr.name}'")
        sample.bools[bitId]
    }
    else -> error("evaluating active condition: unsupported expression $expr")
}
