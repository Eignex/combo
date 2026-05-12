package combo.decisions

import com.eignex.klause.ast.SchemaEntry
import com.eignex.klause.compile.CompiledProblem
import com.eignex.skema.SchemaDef

/**
 * Structural snapshot of a compiled [DecisionSpace]: a klause [CompiledProblem] for the
 * decision side plus the lists of context handles declared on the space. Bandit-family-
 * specific projections (linear feature vectors, tree split tables, …) live in their own
 * modules and consume this as input.
 *
 * This type is intentionally bandit-agnostic. It carries no feature layout, no
 * `encode`/`toObjective` — those concerns belong with the bandit that needs them.
 */
class CompiledDecisionSpace internal constructor(
    val compiled: CompiledProblem,
    val contextBools: List<BoolContextHandle>,
    val contextInts: List<IntContextHandle>,
    val schemaDef: SchemaDef<SchemaEntry>,
)
