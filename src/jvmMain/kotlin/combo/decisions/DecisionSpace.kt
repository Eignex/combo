package combo.decisions

import com.eignex.klause.ast.And
import com.eignex.klause.ast.BoolExpr
import com.eignex.klause.ast.BoolRef
import com.eignex.klause.ast.BoolSpec
import com.eignex.klause.ast.NamedConstraint
import com.eignex.klause.ast.SchemaEntry
import com.eignex.klause.ast.VarSpec
import com.eignex.klause.compile.compile
import kotlin.properties.PropertyDelegateProvider
import kotlin.properties.ReadOnlyProperty

private fun directGateOf(expr: BoolExpr): String = when (expr) {
    is BoolRef -> expr.name
    is And -> directGateOf(expr.children.last())
    else -> error("unexpected activeCondition shape: $expr")
}

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

    /** Identifier serialized as `name` in [DecisionSpaceDef]. Defaults to the subclass's
     *  simple class name; override to a stable string when the JVM name might drift
     *  (e.g. anonymous inner classes). */
    open val name: String get() = this::class.simpleName ?: "DecisionSpace"

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

    /**
     * Optional contextual bool: caller may or may not provide a value at choose time.
     * Allocates a companion `isKnown_<name>` solver variable; when the caller skips
     * `set(handle, …)`, the bandit pins `isKnown_<name>` to false and the value to
     * its default (false), masking the feature slot.
     */
    protected fun optionalContextBool() =
        PropertyDelegateProvider<DecisionSpace, ReadOnlyProperty<DecisionSpace, BoolContextHandle>> { _, prop ->
            val valueName = ctx.qualify(prop.name)
            val gateName = "isKnown_$valueName"
            val gateHandle = ctx.root.registerBool(gateName, ctx.activeCondition)
            val valueHandle = ctx.root.registerBool(
                valueName,
                activeCondition = com.eignex.klause.ast.BoolRef(gateName),
            )
            val h = BoolContextHandle(valueHandle, isKnownGate = gateHandle)
            _contextBools += h
            ReadOnlyProperty { _, _ -> h }
        }

    /**
     * Optional contextual int. See [optionalContextBool] for semantics. When absent the
     * value is pinned to `min` (the domain default).
     */
    protected fun optionalContextInt(min: Int, max: Int) =
        PropertyDelegateProvider<DecisionSpace, ReadOnlyProperty<DecisionSpace, IntContextHandle>> { _, prop ->
            val valueName = ctx.qualify(prop.name)
            val gateName = "isKnown_$valueName"
            val gateHandle = ctx.root.registerBool(gateName, ctx.activeCondition)
            val valueHandle = ctx.root.registerInt(
                valueName, min, max,
                activeCondition = com.eignex.klause.ast.BoolRef(gateName),
            )
            val h = IntContextHandle(valueHandle, isKnownGate = gateHandle)
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
        val def = buildDef()
        SubSpaceContext.clear()
        return def
    }

    private fun buildDef(): DecisionSpaceDef {
        @Suppress("UNCHECKED_CAST")
        val allEntries = ctx.root.definition().entries as Map<String, *>
        // Context variables are klause variables too; tag them so buildLevel doesn't
        // put them in `variables` at the root.
        val contextNames = (_contextBools.map { it.name } + _contextInts.map { it.name }).toSet()
        val rootDef = buildLevel(
            node = this,
            prefix = "",
            allEntries = allEntries,
            ancestorGates = emptyList(),
            excludeNames = contextNames,
        )
        val ctxBools = _contextBools.filter { it.isKnownGate == null }.associate { it.name to (BoolSpec as VarSpec) }
        val ctxInts = _contextInts.filter { it.isKnownGate == null }
            .associate { it.name to (com.eignex.klause.ast.IntSpec(it.min, it.max) as VarSpec) }
        val ctxBoolsOpt = _contextBools.filter { it.isKnownGate != null }.associate { it.name to (BoolSpec as VarSpec) }
        val ctxIntsOpt = _contextInts.filter { it.isKnownGate != null }
            .associate { it.name to (com.eignex.klause.ast.IntSpec(it.min, it.max) as VarSpec) }
        return rootDef.copy(
            name = name,
            context = ctxBools + ctxInts,
            optionalContext = ctxBoolsOpt + ctxIntsOpt,
            interactions = _interactions.map { it.toDef() },
        )
    }

    /**
     * Build a [DecisionSpaceDef] for [node] given the qualifying [prefix]. Recursively
     * walks the [SubSpace.children] tree. Variables/constraints owned at this level are
     * those whose fully-qualified name has no dots after [prefix]. [ancestorGates]
     * carries the chain of optional-sub-space gates above this level so we can
     * recognise variables that are "just regular here" (their `activeCondition` is
     * exactly the ancestor chain) vs. *additionally* optional ones (activeCondition
     * adds an `isKnown_<self>` beyond the ancestor chain).
     */
    private fun buildLevel(
        node: SubSpace,
        prefix: String,
        allEntries: Map<String, *>,
        ancestorGates: List<String>,
        excludeNames: Set<String> = emptySet(),
    ): DecisionSpaceDef {
        val childPrefixes = node.children.map { "$prefix${it.name}." }.toSet()
        val gateNames = node.children.mapNotNull { if (it.isOptional) "$prefix${it.name}" else null }.toSet()
        // Multiples declared at this level: their qualified names sit directly under
        // [prefix] (no further dots in the local name). The constituent bools live at
        // `"<prefix><local>.<label>"` and must not be re-emitted as plain variables.
        val ownMultiples = LinkedHashMap<String, List<String>>()
        val multiplePrefixes = mutableSetOf<String>()
        for ((mName, labels) in ctx.root.multiples) {
            if (!mName.startsWith(prefix)) continue
            val rest = mName.removePrefix(prefix)
            if (rest.contains('.')) continue
            if (childPrefixes.any { mName.startsWith(it) }) continue
            ownMultiples[rest] = labels
            multiplePrefixes += "$mName."
        }
        val ownVariables = LinkedHashMap<String, VarSpec>()
        val ownOptionalVariables = LinkedHashMap<String, VarSpec>()
        val ownConstraints = LinkedHashMap<String, BoolExpr>()

        for ((entryName, entry) in allEntries) {
            if (!entryName.startsWith(prefix)) continue
            val rest = entryName.removePrefix(prefix)
            // Skip entries that belong to a nested child.
            if (childPrefixes.any { entryName.startsWith(it) }) continue
            // Skip the constituent bools of a multiple declared at this level — they
            // round-trip through [DecisionSpaceDef.multiples] instead of `variables`.
            if (multiplePrefixes.any { entryName.startsWith(it) }) continue
            // Skip auto-allocated gate variables — they're regenerated from structure.
            if (entryName in gateNames) continue
            if (entryName in excludeNames) continue
            // Skip the `isKnown_*` gates of optional variables — we recover them from
            // the optional-variable slot instead.
            if (rest.startsWith("isKnown_")) continue
            val localName = rest
            when (entry) {
                is VarSpec -> {
                    val ac = ctx.root.activeConditions[entryName]
                    val gates = ac?.let { gateNamesOf(it) } ?: emptyList()
                    val extra = gates - ancestorGates.toSet()
                    when {
                        extra.isEmpty() -> ownVariables[localName] = entry
                        extra.size == 1 && extra.single() == "isKnown_$entryName" ->
                            ownOptionalVariables[localName] = entry
                        // else: orphaned — fall through silently. Shouldn't happen with
                        // current declarators.
                    }
                }
                is NamedConstraint -> {
                    if (localName.startsWith("__pin_")) continue
                    ownConstraints[localName] = unqualify(entry.expr, prefix)
                }
            }
        }

        val nested = LinkedHashMap<String, DecisionSpaceDef>()
        val optionalNested = LinkedHashMap<String, DecisionSpaceDef>()
        for (child in node.children) {
            val childGates = if (child.isOptional) ancestorGates + "$prefix${child.name}" else ancestorGates
            val childDef = buildLevel(child.space, "$prefix${child.name}.", allEntries, childGates)
            if (child.isOptional) optionalNested[child.name] = childDef else nested[child.name] = childDef
        }

        return DecisionSpaceDef(
            variables = ownVariables,
            optionalVariables = ownOptionalVariables,
            multiples = ownMultiples,
            constraints = ownConstraints,
            spaces = nested,
            optionalSpaces = optionalNested,
        )
    }

    /** Flatten an `And(BoolRef("g1"), BoolRef("g2"))` or single `BoolRef("g")` active
     *  condition into a list of gate variable names. */
    private fun gateNamesOf(expr: BoolExpr): List<String> = when (expr) {
        is BoolRef -> listOf(expr.name)
        is com.eignex.klause.ast.And -> expr.children.flatMap { gateNamesOf(it) }
        else -> emptyList()
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
        val def = buildDef()
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
