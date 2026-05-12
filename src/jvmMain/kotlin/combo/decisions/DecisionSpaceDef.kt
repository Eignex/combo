package combo.decisions

import com.eignex.klause.ast.SchemaEntry
import com.eignex.skema.SchemaDef
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Combo-side serialization counterpart to [CompiledDecisionSpace]. Carries everything
 * needed to reconstruct the bandit-facing schema:
 *
 *  - [klause] — the underlying klause schema (variables, user constraints, auto-
 *    generated pinning constraints for optional sub-spaces). Fully solver-ready as-is.
 *  - [contextBools] / [contextInts] — names of context handles. Klause never sees these;
 *    bandit code reads them at choose/update time via [Context].
 *  - [interactions] — declared cross-feature interactions. Each side is a [ScalarRef]
 *    pointing back into either the klause schema (decisions) or the context list.
 *  - [gates] — dotted names of bool variables that gate an optional sub-space, in
 *    declaration order. Redundant with the `__pin_*` constraints in [klause] but
 *    cheaper to consume.
 */
@Serializable
@SerialName("DecisionSpaceDef")
data class DecisionSpaceDef(
    val klause: SchemaDef<SchemaEntry>,
    val contextBools: List<String>,
    val contextInts: List<String>,
    val interactions: List<InteractionDef>,
    val gates: List<String>,
)

@Serializable
@SerialName("Interaction")
data class InteractionDef(
    val name: String,
    val lhs: ScalarRef,
    val rhs: ScalarRef,
)

/** Tagged reference to one operand of an interaction. */
@Serializable
@SerialName("ScalarRef")
data class ScalarRef(
    val kind: ScalarKind,
    val name: String,
    /** Populated only when [kind] is [ScalarKind.NominalDecision]. */
    val labels: List<String> = emptyList(),
)

@Serializable
enum class ScalarKind {
    BoolDecision, IntDecision, NominalDecision, BoolContext, IntContext,
}
