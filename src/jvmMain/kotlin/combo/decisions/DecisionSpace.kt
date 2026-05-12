package combo.decisions

import com.eignex.klause.ast.SchemaEntry
import com.eignex.klause.compile.compile
import kotlin.properties.PropertyDelegateProvider
import kotlin.properties.ReadOnlyProperty

/**
 * Top-level schema for a bandit. Inherits decision declarators (`boolVar`, `intVar`,
 * `nominal`, `constraint { … }`, `subspace { … }`, `optionalSubspace { … }`) from
 * [SubSpace], and adds declarators for *context* variables and *interactions*.
 *
 * Two ways to consume:
 *  - [definition] — the unified, klause-compatible [DecisionSpaceDef]. Use this when
 *    you only need the schema (e.g. to serialize).
 *  - [compileSpace] — a [CompiledDecisionSpace] that pairs the same definition with a
 *    klause [com.eignex.klause.compile.CompiledProblem] ready for the solver.
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
            val klauseHandle = ctx.root.registerBool(ctx.qualify(prop.name), ctx.activeCondition)
            val h = BoolContextHandle(klauseHandle)
            _contextBools += h
            ReadOnlyProperty { _, _ -> h }
        }

    protected fun contextInt(min: Int, max: Int) =
        PropertyDelegateProvider<DecisionSpace, ReadOnlyProperty<DecisionSpace, IntContextHandle>> { _, prop ->
            val klauseHandle = ctx.root.registerInt(ctx.qualify(prop.name), min, max, ctx.activeCondition)
            val h = IntContextHandle(klauseHandle)
            _contextInts += h
            ReadOnlyProperty { _, _ -> h }
        }

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

    /**
     * The single, fully-serialisable definition of this decision space — variable and
     * constraint entries plus context handles, interactions, and gate names. No solver
     * work involved.
     *
     * Call exactly once per [DecisionSpace] instance: this clears the construction
     * thread-local so a fresh space can be built afterwards. [compileSpace] does the
     * same; they're mutually exclusive entry points.
     */
    fun definition(): DecisionSpaceDef {
        @Suppress("UNCHECKED_CAST")
        val entries = ctx.root.definition().entries as Map<String, SchemaEntry>
        val def = DecisionSpaceDef(
            entries = entries,
            contextBools = _contextBools.map { it.name },
            contextInts = _contextInts.map { it.name },
            interactions = _interactions.map { it.toDef() },
            gates = ctx.root.gates.values.map { it.name },
        )
        SubSpaceContext.clear()
        return def
    }

    /**
     * Compile to a [CompiledDecisionSpace] that pairs [definition] with a solver-ready
     * [com.eignex.klause.compile.CompiledProblem]. Closes the construction context as a
     * side effect so this should be called exactly once per [DecisionSpace] instance;
     * [definition] is the alternative entry point.
     */
    fun compileSpace(): CompiledDecisionSpace {
        val schema = ctx.root
        val compiled = schema.compile()
        @Suppress("UNCHECKED_CAST")
        val entries = schema.definition().entries as Map<String, SchemaEntry>
        val def = DecisionSpaceDef(
            entries = entries,
            contextBools = _contextBools.map { it.name },
            contextInts = _contextInts.map { it.name },
            interactions = _interactions.map { it.toDef() },
            gates = schema.gates.values.map { it.name },
        )
        SubSpaceContext.clear()
        return CompiledDecisionSpace(
            compiled = compiled,
            contextBools = _contextBools.toList(),
            contextInts = _contextInts.toList(),
            interactions = _interactions.toList(),
            definition = def,
            activeConditions = schema.activeConditions.toMap(),
            gates = schema.gates.toMap(),
        )
    }
}
