package combo.bandit.glm

import com.eignex.klause.schema.NominalHandle
import combo.decisions.BoolContextHandle
import combo.decisions.CompiledDecisionSpace
import combo.decisions.IntContextHandle
import combo.decisions.InteractionHandle

/**
 * Linear-bandit-specific feature index over a [CompiledDecisionSpace]. Maps every
 * decision, context, and declared interaction to a stable slot in the dense weight
 * vector that the linear models train against.
 *
 * Layout: `[bool decisions | int decisions | bool contexts | int contexts | interactions]`
 *
 * Interactions that involve a nominal decision expand to one slot per label; all other
 * interactions take a single slot.
 *
 * Trees and other non-linear bandits don't go through this layer — they own their own
 * typed projection over the same [CompiledDecisionSpace].
 */
class LinearFeatureLayout internal constructor(
    val numBoolDecisions: Int,
    val numIntDecisions: Int,
    val boolContexts: List<BoolContextHandle>,
    val intContexts: List<IntContextHandle>,
    val interactions: List<InteractionHandle>,
) {
    val boolDecisionsStart: Int = 0
    val intDecisionsStart: Int = numBoolDecisions
    val boolContextsStart: Int = numBoolDecisions + numIntDecisions
    val intContextsStart: Int = boolContextsStart + boolContexts.size
    val interactionsStart: Int = intContextsStart + intContexts.size

    internal val interactionSlotCount: Map<InteractionHandle, Int> =
        interactions.associateWith { interaction ->
            val nominalSide = interaction.nominalSide()
            nominalSide?.labels?.size ?: 1
        }

    internal val interactionStart: Map<InteractionHandle, Int> = run {
        val out = mutableMapOf<InteractionHandle, Int>()
        var cursor = interactionsStart
        for (h in interactions) {
            out[h] = cursor
            cursor += interactionSlotCount.getValue(h)
        }
        out
    }

    val featureSize: Int =
        interactionsStart + (interactionSlotCount.values.sum())

    internal val boolContextIndex: Map<BoolContextHandle, Int> =
        boolContexts.withIndex().associate { (i, h) -> h to (boolContextsStart + i) }
    internal val intContextIndex: Map<IntContextHandle, Int> =
        intContexts.withIndex().associate { (i, h) -> h to (intContextsStart + i) }

    companion object {
        fun from(space: CompiledDecisionSpace): LinearFeatureLayout = LinearFeatureLayout(
            numBoolDecisions = space.compiled.problem.numBoolVars,
            numIntDecisions = space.compiled.problem.numIntVars,
            boolContexts = space.contextBools,
            intContexts = space.contextInts,
            interactions = space.interactions,
        )
    }
}

internal fun InteractionHandle.nominalSide(): NominalHandle? = when {
    lhs is NominalHandle -> lhs
    rhs is NominalHandle -> rhs
    else -> null
}
