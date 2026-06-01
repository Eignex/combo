package combo.decisions

import com.eignex.klause.ast.And
import com.eignex.klause.ast.AtLeast
import com.eignex.klause.ast.AtMost
import com.eignex.klause.ast.BoolExpr
import com.eignex.klause.ast.BoolRef
import com.eignex.klause.ast.BoolSpec
import com.eignex.klause.ast.CardinalityExpr
import com.eignex.klause.ast.Iff
import com.eignex.klause.ast.Implies
import com.eignex.klause.ast.IntAbs
import com.eignex.klause.ast.IntCmpOp
import com.eignex.klause.ast.IntCompare
import com.eignex.klause.ast.IntExpr
import com.eignex.klause.ast.IntIfThenElse
import com.eignex.klause.ast.IntLit
import com.eignex.klause.ast.IntMax
import com.eignex.klause.ast.IntMin
import com.eignex.klause.ast.IntRef
import com.eignex.klause.ast.IntScale
import com.eignex.klause.ast.IntSpec
import com.eignex.klause.ast.IntSum
import com.eignex.klause.ast.NamedConstraint
import com.eignex.klause.ast.NominalEq
import com.eignex.klause.ast.NominalSpec
import com.eignex.klause.ast.Not
import com.eignex.klause.ast.Or
import com.eignex.klause.ast.PresenceSpec
import com.eignex.klause.ast.SchemaEntry
import com.eignex.klause.ast.VarSpec
import com.eignex.klause.compile.CompiledProblem
import com.eignex.klause.compile.compile
import com.eignex.klause.schema.VariableSchema
import com.eignex.skema.SchemaDef
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/** Prefix used for auto-allocated `isKnown_<name>` bool gates of optional variables and
 *  optional contexts. Single source of truth — refactors that change the gating shape
 *  must update this and the [DecisionSpaceDef.emit] / [buildLevel] code that pattern-
 *  matches on it. */
internal const val OPTIONAL_GATE_PREFIX: String = "isKnown_"

/** Build the gate name for an optional variable / context named [qualified]. */
internal fun gateNameFor(qualified: String): String = "$OPTIONAL_GATE_PREFIX$qualified"

/**
 * Cloud-native config shape for a complete decision space. Hierarchical: every level
 * carries its own variables, constraints, and nested spaces, mirroring how the author
 * organised the Kotlin DSL.
 *
 * Names inside a nested space are *local* — a constraint that says `!premium` inside
 * `slotA` references `slotA.premium` at the solver level. The compile step
 * [toSchemaDef] / [compile] re-qualifies names on the way to klause.
 *
 * Top-level (root) fields:
 *  - [name]: stable identifier carrying through the round-trip.
 *  - [context] / [optionalContext]: caller-supplied values at bandit choose/update.
 *  - [interactions]: declared cross-feature interaction features.
 *
 * Nested levels omit those three. Every level has:
 *  - [variables]: solver variables declared at this level. Keys are local names.
 *  - [constraints]: user-declared constraints. Expression refs use local names.
 *  - [spaces]: always-on nested spaces.
 *  - [optionalSpaces]: gated nested spaces. The compiler synthesises a bool
 *    "gate" variable at the parent level and pinning constraints for everything
 *    inside; neither appears in the JSON.
 */
@Serializable
@SerialName("DecisionSpaceDef")
data class DecisionSpaceDef(
    val name: String? = null,
    val variables: Map<String, VarSpec> = emptyMap(),
    /** Single-variable optionals: each gets an auto-allocated bool gate named
     *  `isKnown_<localName>` and a pin to its default value when the gate is off. */
    val optionalVariables: Map<String, VarSpec> = emptyMap(),
    /** Multi-select variables (`multiple("a", "b", "c")` in the DSL). Each entry
     *  expands at compile time to N bool indicators named `<key>.<label>`. */
    val multiples: Map<String, List<String>> = emptyMap(),
    val constraints: Map<String, BoolExpr> = emptyMap(),
    val spaces: Map<String, DecisionSpaceDef> = emptyMap(),
    val optionalSpaces: Map<String, DecisionSpaceDef> = emptyMap(),
    val context: Map<String, VarSpec> = emptyMap(),
    val optionalContext: Map<String, VarSpec> = emptyMap(),
    val interactions: List<InteractionDef> = emptyList(),
) {
    /** Flatten to klause's [SchemaDef]: qualify every name with its path prefix, splice
     *  user constraints in, synthesise auto-pinning constraints for optional spaces. */
    fun toSchemaDef(): SchemaDef<SchemaEntry> {
        val out = LinkedHashMap<String, SchemaEntry>()
        emit(out, prefix = "")
        return SchemaDef(out)
    }

    /** Compile via klause. */
    fun compile(): CompiledProblem {
        val schema = SchemaLoader()
        val entries = LinkedHashMap<String, SchemaEntry>()
        emit(entries, prefix = "")
        for ((entryName, entry) in entries) schema.addRaw(entryName, entry)
        return schema.compile()
    }

    private fun emit(out: MutableMap<String, SchemaEntry>, prefix: String) {
        for ((localName, spec) in variables) {
            out[prefix + localName] = spec
        }
        for ((localName, labels) in multiples) {
            val qualified = prefix + localName
            for (label in labels) out["$qualified.$label"] = BoolSpec
        }
        for ((localName, spec) in optionalVariables) emitOptional(out, prefix + localName, spec)
        for ((localName, spec) in context) {
            out[prefix + localName] = spec
        }
        for ((localName, spec) in optionalContext) emitOptional(out, prefix + localName, spec)
        for ((localName, expr) in constraints) {
            out[prefix + localName] = NamedConstraint(qualify(expr, prefix))
        }
        for ((localName, child) in spaces) {
            child.emit(out, prefix = "$prefix$localName.")
        }
        for ((localName, child) in optionalSpaces) {
            val gateName = prefix + localName
            out[gateName] = BoolSpec
            val childPrefix = "$prefix$localName."
            // Emit the child into a scratch map so we can scan exactly the entries it
            // produced — no slicing of the (potentially huge) parent map.
            val childOut = LinkedHashMap<String, SchemaEntry>()
            child.emit(childOut, prefix = childPrefix)
            for ((entryName, entry) in childOut) {
                out[entryName] = entry
                if (entry is VarSpec && !entryName.startsWith(OPTIONAL_GATE_PREFIX)) {
                    out["__pin_$entryName"] = NamedConstraint(synthesizePin(entryName, entry, gateName))
                }
            }
        }
    }

    private fun emitOptional(out: MutableMap<String, SchemaEntry>, qualified: String, spec: VarSpec) {
        val gateName = gateNameFor(qualified)
        // The gate is a klause PresenceSpec naming its value var, so klause's own absent-value
        // pinning fixes the value to its default when the gate is false — no combo `__pin_`.
        out[gateName] = PresenceSpec(qualified)
        out[qualified] = spec
    }
}

@Serializable
@SerialName("Interaction")
data class InteractionDef(
    val name: String,
    val lhs: ScalarRef,
    val rhs: ScalarRef,
)

/**
 * Reference to one side of an interaction. The path navigates the decision-space tree:
 *  - `[ "context", "premiumCtx" ]` → the root's `context["premiumCtx"]`
 *  - `[ "slotA", "budget" ]` → `spaces["slotA"].variables["budget"]`
 *  - `[ "slotA", "audio", "mute" ]` → nested into an `optionalSpaces["audio"]`
 *
 * The last segment is the leaf variable / context name; earlier segments are nested
 * decision-space names. The wire format uses local-name segments only.
 */
@Serializable
@SerialName("ScalarRef")
data class ScalarRef(
    val kind: ScalarKind,
    val path: List<String>,
    /** Populated only when [kind] is [ScalarKind.NominalDecision] or
     *  [ScalarKind.NominalContext]. */
    val labels: List<String> = emptyList(),
) {
    init {
        require(path.isNotEmpty()) { "ScalarRef path must be non-empty" }
    }
}

@Serializable
enum class ScalarKind {
    BoolDecision, IntDecision, NominalDecision,
    BoolContext, IntContext, NominalContext,
}

/** Internal helper: build a [VariableSchema] from a pre-existing entries map. */
internal class SchemaLoader : VariableSchema() {
    fun addRaw(name: String, entry: SchemaEntry) = add(name, entry)
}

/**
 * Default-value pin for a variable whose gate is off:
 *  - bool   → false
 *  - int    → domain min
 *  - nominal → first label
 */
internal fun synthesizePin(qualifiedName: String, spec: VarSpec, gateName: String): BoolExpr {
    val notGate = Not(BoolRef(gateName))
    return when (spec) {
        is PresenceSpec ->
            // A presence marker is a gate, never a pinnable value. The synthesizePin path is
            // only used for sub-space-gated value vars; gates never reach here.
            throw UnsupportedOperationException("PresenceSpec '$qualifiedName' is a gate, not a value")
        is BoolSpec -> Implies(notGate, Not(BoolRef(qualifiedName)))
        is IntSpec -> Implies(notGate, IntCompare(IntRef(qualifiedName), IntCmpOp.EQ, IntLit(spec.min)))
        is NominalSpec -> Implies(notGate, NominalEq(qualifiedName, spec.labels[0]))
        is com.eignex.klause.ast.FloatSpec ->
            // Pin the underlying bucket-int to 0 → real value is `min` (the float's lower bound),
            // mirroring the int convention.
            Implies(notGate, IntCompare(IntRef(qualifiedName), IntCmpOp.EQ, IntLit(0)))
        is com.eignex.klause.ast.SetSpec, is com.eignex.klause.ast.MultipleSpec ->
            // Set/multiple variables are not exposed as optional gated variables by combo
            // yet (multi-selects flow through the dedicated `multiples` map, not VarSpec).
            // When that migration lands the gated-off default is "empty set" — assert no
            // indicator is selected. Guarded explicitly so the omission is loud, not silent.
            throw UnsupportedOperationException(
                "optional set/multiple variable '$qualifiedName' is not supported yet",
            )
    }
}

// -----------------------------------------------------------------------------
// Expression name rewriting — strips a qualifying prefix when emitting (so JSON
// inside a subspace says `!premium` not `!slotA.premium`) and re-adds it on the
// way back to klause.
// -----------------------------------------------------------------------------

/** Walk [expr] and replace every name with `prefix + name`. */
internal fun qualify(expr: BoolExpr, prefix: String): BoolExpr {
    if (prefix.isEmpty()) return expr
    return rewriteBool(expr) { name -> prefix + name }
}

/** Walk [expr] and strip [prefix] from any name that starts with it. */
internal fun unqualify(expr: BoolExpr, prefix: String): BoolExpr {
    if (prefix.isEmpty()) return expr
    return rewriteBool(expr) { name -> if (name.startsWith(prefix)) name.removePrefix(prefix) else name }
}

private fun rewriteBool(expr: BoolExpr, transform: (String) -> String): BoolExpr = when (expr) {
    is BoolRef -> BoolRef(transform(expr.name), expr.negated)
    is NominalEq -> NominalEq(transform(expr.name), expr.label)
    is Not -> Not(rewriteBool(expr.child, transform))
    is And -> And(expr.children.map { rewriteBool(it, transform) })
    is Or -> Or(expr.children.map { rewriteBool(it, transform) })
    is Implies -> Implies(rewriteBool(expr.left, transform), rewriteBool(expr.right, transform))
    is Iff -> Iff(rewriteBool(expr.left, transform), rewriteBool(expr.right, transform))
    is AtMost -> AtMost(expr.children.map { rewriteBool(it, transform) }, expr.k)
    is AtLeast -> AtLeast(expr.children.map { rewriteBool(it, transform) }, expr.k)
    is CardinalityExpr -> CardinalityExpr(expr.children.map { rewriteBool(it, transform) }, expr.min, expr.max)
    is IntCompare -> IntCompare(rewriteInt(expr.left, transform), expr.op, rewriteInt(expr.right, transform))
    else -> expr  // AllDifferent, TableConstraint, XorExpr, PseudoBoolean: not yet exercised; pass-through
}

private fun rewriteInt(expr: IntExpr, transform: (String) -> String): IntExpr = when (expr) {
    is IntRef -> IntRef(transform(expr.name))
    is IntLit -> expr
    is IntScale -> IntScale(expr.coeff, rewriteInt(expr.child, transform))
    is IntSum -> IntSum(expr.children.map { rewriteInt(it, transform) })
    is IntMin -> IntMin(expr.children.map { rewriteInt(it, transform) })
    is IntMax -> IntMax(expr.children.map { rewriteInt(it, transform) })
    is IntAbs -> IntAbs(rewriteInt(expr.child, transform))
    is IntIfThenElse -> IntIfThenElse(
        rewriteBool(expr.cond, transform),
        rewriteInt(expr.thenE, transform),
        rewriteInt(expr.elseE, transform),
    )
    else -> expr  // IntElement / IntMul / IntDiv / IntMod: pass-through until exercised
}
