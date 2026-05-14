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

    /** Direct children mounted via `decisionSpace` / `optionalDecisionSpace` on this
     *  space. Drives the hierarchical [DecisionSpaceDef] emission. */
    internal val children: MutableList<ChildSpace> = mutableListOf()

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

    protected fun floatVar(min: Double, max: Double, buckets: Int = 32) =
        PropertyDelegateProvider<SubSpace, ReadOnlyProperty<SubSpace, FloatHandle>> { _, prop ->
            val handle = ctx.root.registerFloat(ctx.qualify(prop.name), min, max, buckets, ctx.activeCondition)
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
     * Mount a typed nested decision space. The factory runs with a derived context
     * whose prefix is `"${currentPrefix}.${propertyName}"`, so every declaration
     * inside the factory registers under the qualified namespace.
     */
    protected fun <T : SubSpace> decisionSpace(factory: () -> T) =
        PropertyDelegateProvider<SubSpace, ReadOnlyProperty<SubSpace, T>> { thisRef, prop ->
            val instance = SubSpaceContext.withContext(ctx.child(prop.name), factory)
            thisRef.children += ChildSpace(prop.name, instance, gate = null)
            ReadOnlyProperty { _, _ -> instance }
        }

    /**
     * Mount a typed nested decision space *gated* by an auto-allocated bool variable.
     * Every variable declared inside the factory is pinned to its default (false /
     * domain minimum / first nominal label) when the gate is off, via a reified klause
     * constraint synthesised at compile time.
     *
     * Nested optional spaces compose: a variable two levels deep is active only when
     * both gates are on.
     */
    protected fun <T : SubSpace> optionalDecisionSpace(factory: () -> T) =
        PropertyDelegateProvider<SubSpace, ReadOnlyProperty<SubSpace, T>> { thisRef, prop ->
            val gateName = ctx.qualify(prop.name)
            val gateHandle = ctx.root.registerBool(gateName, ctx.activeCondition)
            val instance = SubSpaceContext.withContext(ctx.gatedChild(prop.name, gateName), factory)
            ctx.root.recordGate(instance, gateHandle)
            thisRef.children += ChildSpace(prop.name, instance, gate = gateHandle)
            ReadOnlyProperty { _, _ -> instance }
        }

    @Deprecated("Renamed to decisionSpace", ReplaceWith("decisionSpace(factory)"))
    protected fun <T : SubSpace> subspace(factory: () -> T) = decisionSpace(factory)

    @Deprecated("Renamed to optionalDecisionSpace", ReplaceWith("optionalDecisionSpace(factory)"))
    protected fun <T : SubSpace> optionalSubspace(factory: () -> T) = optionalDecisionSpace(factory)
}

/** Internal: parent's record of a nested decision space and (if optional) its gate. */
internal class ChildSpace(
    val name: String,
    val space: SubSpace,
    val gate: com.eignex.klause.schema.BoolHandle?,
) {
    val isOptional: Boolean get() = gate != null
}
