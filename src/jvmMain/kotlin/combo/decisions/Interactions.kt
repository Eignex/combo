package combo.decisions

import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.IntHandle
import com.eignex.klause.schema.NominalHandle

/**
 * A declared product feature `lhs × rhs`. Bandit-family projections layer interactions
 * on top of the basic feature layout — linear bandits get an extra weight slot per
 * interaction, trees ignore them since they discover interactions natively.
 *
 * Allowed pairings (validated at registration time, see [DecisionSpace.interact]):
 *  - `context × decision` (the product becomes a linear term in the decision at
 *    choose-time, with the context value folded into the coefficient).
 *  - `context × context` (the product is a constant during a choose call).
 *
 *  Decision × decision interactions would need a quadratic objective in klause and are
 *  not supported in this revision.
 */
class InteractionHandle internal constructor(
    val name: String,
    internal val lhs: Any,
    internal val rhs: Any,
) {
    internal val involvesDecision: Boolean = lhs.isDecision() || rhs.isDecision()
}

internal fun Any.isDecision(): Boolean = this is BoolHandle || this is IntHandle || this is NominalHandle
internal fun Any.isContext(): Boolean = this is BoolContextHandle || this is IntContextHandle

internal fun isAllowedInteraction(lhs: Any, rhs: Any): Boolean {
    val l = lhs.isDecision() || lhs.isContext()
    val r = rhs.isDecision() || rhs.isContext()
    if (!l || !r) return false
    // Decision × decision is disallowed (quadratic objective).
    return !(lhs.isDecision() && rhs.isDecision())
}
