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
 * the inherited declarators, then mount instances via [submodel] on a parent.
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
            val handle = ctx.root.registerBool(ctx.qualify(prop.name))
            ReadOnlyProperty { _, _ -> handle }
        }

    protected fun intVar(min: Int, max: Int) =
        PropertyDelegateProvider<SubSpace, ReadOnlyProperty<SubSpace, IntHandle>> { _, prop ->
            val handle = ctx.root.registerInt(ctx.qualify(prop.name), min, max)
            ReadOnlyProperty { _, _ -> handle }
        }

    protected fun nominal(vararg labels: String) =
        PropertyDelegateProvider<SubSpace, ReadOnlyProperty<SubSpace, NominalHandle>> { _, prop ->
            val handle = ctx.root.registerNominal(ctx.qualify(prop.name), labels.toList())
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
    protected fun <T : SubSpace> submodel(factory: () -> T) =
        PropertyDelegateProvider<SubSpace, ReadOnlyProperty<SubSpace, T>> { _, prop ->
            val instance = SubSpaceContext.withContext(ctx.child(prop.name)) { factory() }
            ReadOnlyProperty { _, _ -> instance }
        }
}
