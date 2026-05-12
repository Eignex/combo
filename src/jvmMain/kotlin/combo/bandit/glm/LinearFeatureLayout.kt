package combo.bandit.glm

import combo.decisions.BoolContextHandle
import combo.decisions.CompiledDecisionSpace
import combo.decisions.IntContextHandle
import combo.decisions.InteractionHandle

/**
 * Linear-bandit-specific feature index over a [CompiledDecisionSpace]. Maps every
 * decision, context, and declared interaction to a stable slot in the dense weight
 * vector that the linear models train against.
 *
 * Layout:
 *   `[bool decisions | int decisions | bool contexts | int contexts | interactions]`
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

    val featureSize: Int = interactionsStart + interactions.size

    internal val boolContextIndex: Map<BoolContextHandle, Int> =
        boolContexts.withIndex().associate { (i, h) -> h to (boolContextsStart + i) }
    internal val intContextIndex: Map<IntContextHandle, Int> =
        intContexts.withIndex().associate { (i, h) -> h to (intContextsStart + i) }
    internal val interactionIndex: Map<InteractionHandle, Int> =
        interactions.withIndex().associate { (i, h) -> h to (interactionsStart + i) }

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
