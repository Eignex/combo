package combo.decisions

import com.eignex.klause.ast.BoolExpr
import com.eignex.klause.ast.BoolSpec
import com.eignex.klause.ast.IntSpec
import com.eignex.klause.ast.NamedConstraint
import com.eignex.klause.ast.NominalSpec
import com.eignex.klause.ast.SchemaEntry
import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.IntHandle
import com.eignex.klause.schema.NominalHandle
import com.eignex.klause.schema.VariableSchema

/**
 * Construction-time wiring for [SubSpace]. Threads the current dotted prefix
 * down through nested sub-model factories so each `boolVar()` / `intVar()` /
 * `constraint { ... }` registers with the *root* klause schema under a
 * fully-qualified name.
 *
 * Lives on a thread-local — the root [DecisionSpace] installs one before its
 * subclass property initializers run, and child sub-models swap in a derived
 * context (same root, deeper prefix) for the duration of their factory call.
 */
internal class SubSpaceContext private constructor(
    val root: RootKlauseSchema,
    val prefix: String,
) {
    fun qualify(name: String): String = if (prefix.isEmpty()) name else "$prefix.$name"
    fun child(propertyName: String): SubSpaceContext =
        SubSpaceContext(root, qualify(propertyName))

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

        fun makeRoot(): SubSpaceContext = SubSpaceContext(RootKlauseSchema(), "")

        /** Run [block] with [ctx] as the current context, restoring whatever was set before. */
        fun <T> withContext(ctx: SubSpaceContext, block: () -> T): T {
            val prev = threadLocal.get()
            threadLocal.set(ctx)
            try {
                return block()
            } finally {
                threadLocal.set(prev)
            }
        }

        /** Clear the thread-local. Called by [DecisionSpace.compileSpace] when construction is done. */
        fun clear() {
            threadLocal.set(null)
        }
    }
}

/**
 * Klause [VariableSchema] wrapper that exposes the protected `add` method to
 * [combo.decisions] so qualified-name registrations from sub-models can be
 * forwarded into the single root klause schema.
 */
internal class RootKlauseSchema : VariableSchema() {
    fun registerBool(name: String): BoolHandle {
        add(name, BoolSpec)
        return BoolHandle(name)
    }

    fun registerInt(name: String, min: Int, max: Int): IntHandle {
        add(name, IntSpec(min, max))
        return IntHandle(name, min, max)
    }

    fun registerNominal(name: String, labels: List<String>): NominalHandle {
        add(name, NominalSpec(labels))
        return NominalHandle(name, labels)
    }

    fun registerConstraint(name: String, expr: BoolExpr) {
        add(name, NamedConstraint(expr))
    }

    @Suppress("UNCHECKED_CAST")
    fun entries(): Map<String, SchemaEntry> = (definition().entries as Map<String, SchemaEntry>)
}
