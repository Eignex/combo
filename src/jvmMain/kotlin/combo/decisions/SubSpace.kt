package combo.decisions

import com.eignex.klause.ast.BoolExpr
import com.eignex.klause.ast.NamedConstraint
import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.FloatHandle
import com.eignex.klause.schema.IntHandle
import com.eignex.klause.schema.NominalHandle
import kotlin.properties.PropertyDelegateProvider
import kotlin.properties.ReadOnlyProperty

/**
 * Base class for typed, reusable sub-schemas mounted under a [DecisionSpace] (or under
 * another sub-space, transitively). Subclass it, declare variables and constraints with
 * the inherited declarators, then mount instances via [subspace] on a parent.
 *
 * Construction relies on a thread-local "current context" installed by the root
 * [DecisionSpace] — every `boolVar()` etc. registers with the *same* root klause schema,
 * but under a fully-qualified, dotted name like `"codeA.foo"`. The user-facing handles
 * (e.g. `outer.codeA.foo`) carry those qualified names directly, so decoding a klause
 * `Sample` against them works without any indirection.
 */
abstract class SubSpace internal constructor() {

    internal val ctx: SubSpaceContext = SubSpaceContext.installOrCurrent {
        SubSpaceContext.makeRoot()
    }

    protected fun boolVar() =
        PropertyDelegateProvider<SubSpace, ReadOnlyProperty<SubSpace, BoolHandle>> { _, prop ->
            val handle = ctx.root.registerBool(ctx.qualify(prop.name), ctx.activeCondition)
            ReadOnlyProperty { _, _ -> handle }
        }

    protected fun intVar(min: Int, max: Int) =
        PropertyDelegateProvider<SubSpace, ReadOnlyProperty<SubSpace, IntHandle>> { _, prop ->
            val handle = ctx.root.registerInt(ctx.qualify(prop.name), min, max, ctx.activeCondition)
            ReadOnlyProperty { _, _ -> handle }
        }

    protected fun nominal(vararg labels: String) =
        PropertyDelegateProvider<SubSpace, ReadOnlyProperty<SubSpace, NominalHandle>> { _, prop ->
            val handle = ctx.root.registerNominal(ctx.qualify(prop.name), labels.toList(), ctx.activeCondition)
            ReadOnlyProperty { _, _ -> handle }
        }

    protected fun constraint(build: () -> BoolExpr) =
        PropertyDelegateProvider<SubSpace, ReadOnlyProperty<SubSpace, NamedConstraint>> { _, prop ->
            val name = ctx.qualify(prop.name)
            val nc = NamedConstraint(build())
            ctx.root.registerConstraint(name, nc.expr)
            ReadOnlyProperty { _, _ -> nc }
        }

    /**
     * Mount a typed sub-space as a property on this space. The factory runs with a
     * derived context whose prefix is `"${currentPrefix}.${propertyName}"`, so every
     * declaration inside the factory registers under the qualified namespace.
     */
    protected fun <T : SubSpace> subspace(factory: () -> T) =
        PropertyDelegateProvider<SubSpace, ReadOnlyProperty<SubSpace, T>> { _, prop ->
            val instance = SubSpaceContext.withContext(ctx.child(prop.name), factory)
            ReadOnlyProperty { _, _ -> instance }
        }

    /**
     * Mount a typed sub-space *gated* by an auto-allocated bool variable. The gate's
     * klause name is the property name itself; the sub-space body sits under that
     * namespace. Every variable declared inside the factory is pinned to a default
     * (false / domain minimum / first nominal label) when the gate is off, via a
     * reified klause constraint added at compile time.
     *
     * Nested optional sub-spaces compose activation conditions — a variable two levels
     * deep is active only when both gates are on.
     */
    protected fun <T : SubSpace> optionalSubspace(factory: () -> T) =
        PropertyDelegateProvider<SubSpace, ReadOnlyProperty<SubSpace, T>> { _, prop ->
            val gateName = ctx.qualify(prop.name)
            // Register the gate first, under the *parent's* activation. If we're
            // already inside an optional, the gate itself becomes optional too — it
            // gets pinned to false when the outer gate is off, which is correct.
            val gateHandle = ctx.root.registerBool(gateName, ctx.activeCondition)
            val instance = SubSpaceContext.withContext(ctx.gatedChild(prop.name, gateName), factory)
            ctx.root.recordGate(instance, gateHandle)
            ReadOnlyProperty { _, _ -> instance }
        }
}
