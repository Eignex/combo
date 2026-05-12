package combo.bandit.glm

import com.eignex.klause.schema.NominalHandle
import combo.decisions.CompiledDecisionSpace
import combo.decisions.InteractionHandle

/**
 * Linear-bandit-specific feature index over a [CompiledDecisionSpace]. Maps every
 * solver variable (decisions and contexts alike) plus declared interactions to a stable
 * slot in the dense weight vector that the linear models train against.
 *
 * Layout: `[all bool vars | all int vars | interactions]`
 *
 * Contexts no longer get their own block — they are solver variables now, pinned per
 * call via [com.eignex.klause.solver.Assumptions]. Interactions involving a nominal
 * decision expand to one slot per label.
 */
class LinearFeatureLayout internal constructor(
    val numBoolVars: Int,
    val numIntVars: Int,
    val interactions: List<InteractionHandle>,
) {
    val boolStart: Int = 0
    val intStart: Int = numBoolVars
    val interactionsStart: Int = numBoolVars + numIntVars

    internal val interactionSlotCount: Map<InteractionHandle, Int> =
        interactions.associateWith { it.nominalSide()?.labels?.size ?: 1 }

    internal val interactionStart: Map<InteractionHandle, Int> = run {
        val out = mutableMapOf<InteractionHandle, Int>()
        var cursor = interactionsStart
        for (h in interactions) {
            out[h] = cursor
            cursor += interactionSlotCount.getValue(h)
        }
        out
    }

    val featureSize: Int = interactionsStart + interactionSlotCount.values.sum()

    companion object {
        fun from(space: CompiledDecisionSpace): LinearFeatureLayout = LinearFeatureLayout(
            numBoolVars = space.compiled.problem.numBoolVars,
            numIntVars = space.compiled.problem.numIntVars,
            interactions = space.interactions,
        )
    }
}

internal fun InteractionHandle.nominalSide(): NominalHandle? = when {
    lhs is NominalHandle -> lhs
    rhs is NominalHandle -> rhs
    else -> null
}
