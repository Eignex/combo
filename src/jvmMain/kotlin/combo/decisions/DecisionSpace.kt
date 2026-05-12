package combo.decisions

import com.eignex.klause.ast.SchemaEntry
import com.eignex.klause.compile.compile
import com.eignex.skema.SchemaDef
import kotlin.properties.PropertyDelegateProvider
import kotlin.properties.ReadOnlyProperty

/**
 * Top-level schema for a bandit. Inherits decision declarators (`boolVar`, `intVar`,
 * `nominal`, `constraint { … }`, `subspace { … }`) from [SubSpace], and adds
 * declarators for *context* variables — values the caller supplies at every
 * choose/update call rather than letting the bandit pick.
 *
 * Compile the space with [compileSpace] to get a [CompiledDecisionSpace]; that's the
 * structural artifact the bandit-family projections (e.g.
 * [combo.bandit.glm.LinearFeatureProjection]) build on.
 */
abstract class DecisionSpace : SubSpace() {

    private val _contextBools = mutableListOf<BoolContextHandle>()
    private val _contextInts = mutableListOf<IntContextHandle>()
    private val _interactions = mutableListOf<InteractionHandle>()

    internal val contextBools: List<BoolContextHandle> get() = _contextBools
    internal val contextInts: List<IntContextHandle> get() = _contextInts
    internal val interactions: List<InteractionHandle> get() = _interactions

    protected fun contextBool() =
        PropertyDelegateProvider<DecisionSpace, ReadOnlyProperty<DecisionSpace, BoolContextHandle>> { _, prop ->
            val h = BoolContextHandle(prop.name)
            _contextBools += h
            ReadOnlyProperty { _, _ -> h }
        }

    protected fun contextInt() =
        PropertyDelegateProvider<DecisionSpace, ReadOnlyProperty<DecisionSpace, IntContextHandle>> { _, prop ->
            val h = IntContextHandle(prop.name)
            _contextInts += h
            ReadOnlyProperty { _, _ -> h }
        }

    /**
     * Declare a product feature combining two scalar handles. Supported pairings:
     *  - `context × decision` (bool or int on either side)
     *  - `context × context`
     *
     * Decision × decision is rejected at registration time — that would require a
     * quadratic objective which klause's [com.eignex.klause.solver.LinearObjective]
     * doesn't express.
     *
     * Bandit-family projections decide whether to materialise the interaction:
     * [combo.bandit.glm.LinearFeatureProjection] adds an extra weight slot, trees
     * ignore the declaration.
     */
    protected fun interact(lhs: Any, rhs: Any) =
        PropertyDelegateProvider<DecisionSpace, ReadOnlyProperty<DecisionSpace, InteractionHandle>> { _, prop ->
            require(isAllowedInteraction(lhs, rhs)) {
                "Unsupported interaction at '${prop.name}': lhs=$lhs rhs=$rhs. " +
                    "Allowed pairings are context×decision and context×context."
            }
            val h = InteractionHandle(prop.name, lhs, rhs)
            _interactions += h
            ReadOnlyProperty { _, _ -> h }
        }

    /** Compile to a structural [CompiledDecisionSpace]. Bandit-specific projections
     *  (e.g. [combo.bandit.glm.LinearFeatureProjection]) layer on top of this. */
    fun compileSpace(): CompiledDecisionSpace {
        val schema = ctx.root
        val klauseEntries: SchemaDef<SchemaEntry> = schema.definition()
        val compiled = schema.compile()
        SubSpaceContext.clear()
        return CompiledDecisionSpace(
            compiled = compiled,
            contextBools = _contextBools.toList(),
            contextInts = _contextInts.toList(),
            schemaDef = klauseEntries,
            activeConditions = schema.activeConditions.toMap(),
            gates = schema.gates.toMap(),
            interactions = _interactions.toList(),
        )
    }
}
