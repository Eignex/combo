package combo.decisions

import com.eignex.klause.ast.And
import com.eignex.klause.ast.BoolExpr
import com.eignex.klause.ast.BoolRef
import com.eignex.klause.ast.BoolSpec
import com.eignex.klause.ast.FloatSpec
import com.eignex.klause.ast.Implies
import com.eignex.klause.ast.IntCmpOp
import com.eignex.klause.ast.IntCompare
import com.eignex.klause.ast.IntLit
import com.eignex.klause.ast.IntRef
import com.eignex.klause.ast.IntSpec
import com.eignex.klause.ast.NamedConstraint
import com.eignex.klause.ast.NominalEq
import com.eignex.klause.ast.NominalSpec
import com.eignex.klause.ast.Not
import com.eignex.klause.ast.SchemaEntry
import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.FloatHandle
import com.eignex.klause.schema.IntHandle
import com.eignex.klause.schema.NominalHandle
import com.eignex.klause.schema.VariableSchema

/**
 * Construction-time wiring for [SubSpace]. Threads the current dotted prefix and
 * activation condition down through nested sub-space factories so each `boolVar()` /
 * `intVar()` / `constraint { ... }` registers with the *root* klause schema under a
 * fully-qualified name, tagged with whichever gate(s) make it conditionally present.
 *
 * Lives on a thread-local — the root [DecisionSpace] installs one before its subclass
 * property initializers run, and child sub-spaces swap in a derived context for the
 * duration of their factory call.
 */
internal class SubSpaceContext private constructor(
    val root: RootKlauseSchema,
    val prefix: String,
    val activeCondition: BoolExpr?,
) {
    fun qualify(name: String): String = if (prefix.isEmpty()) name else "$prefix.$name"

    /** Always-active child context, used by plain `decisionSpace { ... }`. */
    fun child(propertyName: String): SubSpaceContext =
        SubSpaceContext(root, qualify(propertyName), activeCondition)

    /** Child context gated by a freshly-registered bool variable. Combines AND-wise
     *  with whatever activation condition the parent already had. */
    fun gatedChild(propertyName: String, gateName: String): SubSpaceContext {
        val gateExpr: BoolExpr = BoolRef(gateName)
        val composed = activeCondition?.let { And(listOf(it, gateExpr)) } ?: gateExpr
        return SubSpaceContext(root, qualify(propertyName), composed)
    }

    companion object {
        private val threadLocal = ThreadLocal<SubSpaceContext?>()

        /** Read the current context, installing a fresh root if there isn't one. */
        fun installOrCurrent(makeRoot: () -> SubSpaceContext): SubSpaceContext {
            val existing = threadLocal.get()
            if (existing != null) return existing
            val fresh = makeRoot()
            threadLocal.set(fresh)
            return fresh
        }

        fun makeRoot(): SubSpaceContext = SubSpaceContext(RootKlauseSchema(), "", activeCondition = null)

        fun <T> withContext(ctx: SubSpaceContext, block: () -> T): T {
            val prev = threadLocal.get()
            threadLocal.set(ctx)
            try {
                return block()
            } finally {
                threadLocal.set(prev)
            }
        }

        fun clear() {
            threadLocal.set(null)
        }
    }
}

/**
 * Klause [VariableSchema] wrapper that exposes the protected `add` method to
 * [combo.decisions] and pins optional variables to default values when their gate is
 * off. Also records each variable's activation condition for later projection lookup
 * (linear bandits zero inactive feature slots; tree bandits emit `isPresent` columns).
 */
internal class RootKlauseSchema : VariableSchema() {

    private val _activeConditions = mutableMapOf<String, BoolExpr>()
    val activeConditions: Map<String, BoolExpr> get() = _activeConditions

    private val _gates = mutableMapOf<SubSpace, BoolHandle>()
    val gates: Map<SubSpace, BoolHandle> get() = _gates

    fun recordGate(sub: SubSpace, handle: BoolHandle) {
        _gates[sub] = handle
    }

    /** Multi-select variables, keyed by fully-qualified name. Each value is the ordered
     *  label list; the constituent bool variables are registered as `<name>.<label>`. */
    private val _multiples = LinkedHashMap<String, List<String>>()
    val multiples: Map<String, List<String>> get() = _multiples

    fun registerBool(name: String, activeCondition: BoolExpr?): BoolHandle {
        add(name, BoolSpec)
        if (activeCondition != null) {
            _activeConditions[name] = activeCondition
            // !activeCondition → !var (default = false)
            add("__pin_$name", NamedConstraint(Implies(Not(activeCondition), Not(BoolRef(name)))))
        }
        return BoolHandle(name)
    }

    fun registerInt(name: String, min: Int, max: Int, activeCondition: BoolExpr?): IntHandle {
        add(name, IntSpec(min, max))
        if (activeCondition != null) {
            _activeConditions[name] = activeCondition
            // !activeCondition → var == min (default = lower bound)
            add(
                "__pin_$name",
                NamedConstraint(
                    Implies(
                        Not(activeCondition),
                        IntCompare(IntRef(name), IntCmpOp.EQ, IntLit(min)),
                    ),
                ),
            )
        }
        return IntHandle(name, min, max)
    }

    fun registerFloat(name: String, min: Double, max: Double, buckets: Int, activeCondition: BoolExpr?): FloatHandle {
        add(name, FloatSpec(min, max, buckets))
        if (activeCondition != null) {
            _activeConditions[name] = activeCondition
            // !activeCondition → var's bucket == 0 (default = lower bound, mirrors int pinning).
            add(
                "__pin_$name",
                NamedConstraint(
                    Implies(
                        Not(activeCondition),
                        IntCompare(IntRef(name), IntCmpOp.EQ, IntLit(0)),
                    ),
                ),
            )
        }
        return FloatHandle(name, min, max, buckets)
    }

    fun registerNominal(name: String, labels: List<String>, activeCondition: BoolExpr?): NominalHandle {
        add(name, NominalSpec(labels))
        if (activeCondition != null) {
            _activeConditions[name] = activeCondition
            // !activeCondition → var == labels[0] (default = first label)
            add(
                "__pin_$name",
                NamedConstraint(Implies(Not(activeCondition), NominalEq(name, labels[0]))),
            )
        }
        return NominalHandle(name, labels)
    }

    fun registerMultiple(name: String, labels: List<String>, activeCondition: BoolExpr?): MultipleHandle {
        require(labels.isNotEmpty()) { "Multiple '$name' must have at least one label" }
        require(labels.distinct().size == labels.size) { "Multiple '$name' has duplicate labels: $labels" }
        _multiples[name] = labels.toList()
        val perLabel = LinkedHashMap<String, BoolHandle>()
        for (label in labels) perLabel[label] = registerBool("$name.$label", activeCondition)
        return MultipleHandle(name, labels.toList(), perLabel)
    }

    fun registerConstraint(name: String, expr: BoolExpr) {
        add(name, NamedConstraint(expr))
    }

    fun entries(): Map<String, SchemaEntry> = definition().entries
}
