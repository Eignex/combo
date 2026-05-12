package combo.decisions

import com.eignex.klause.ast.SchemaEntry
import com.eignex.klause.compile.CompiledProblem
import com.eignex.klause.compile.compile
import com.eignex.klause.schema.VariableSchema
import com.eignex.skema.SchemaDef
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Single serializable record of a complete combo decision space.
 *
 *  - [entries] — every variable / constraint / auto-generated pinning constraint for
 *    optional sub-spaces, keyed by fully-qualified dotted name. Solver-compatible:
 *    project via [toSchemaDef] and feed to a constraint solver.
 *  - [contextBools] / [contextInts] — names of context handles. The solver never sees
 *    these; bandit code reads them at choose/update time via [Context].
 *  - [interactions] — declared cross-feature interactions. Each side is a [ScalarRef]
 *    pointing back into either [entries] (decisions) or the context lists.
 *  - [gates] — dotted names of bool variables that gate an optional sub-space.
 *    Redundant with the `__pin_*` constraints in [entries] but cheaper to consume.
 */
@Serializable
@SerialName("DecisionSpaceDef")
data class DecisionSpaceDef(
    val entries: Map<String, SchemaEntry>,
    val contextBools: List<String> = emptyList(),
    val contextInts: List<String> = emptyList(),
    val interactions: List<InteractionDef> = emptyList(),
    val gates: List<String> = emptyList(),
) {
    /** Project the variable + constraint entries onto the underlying solver's schema
     *  type for direct consumption. */
    fun toSchemaDef(): SchemaDef<SchemaEntry> = SchemaDef(entries)

    /** Compile the constraint side of this schema to a solver-ready [CompiledProblem]. */
    fun compile(): CompiledProblem {
        val schema = SchemaLoader()
        for ((name, entry) in entries) schema.addRaw(name, entry)
        return schema.compile()
    }
}

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

/** Internal helper: build a [VariableSchema] from a pre-existing entries map. */
internal class SchemaLoader : VariableSchema() {
    fun addRaw(name: String, entry: SchemaEntry) = add(name, entry)
}
