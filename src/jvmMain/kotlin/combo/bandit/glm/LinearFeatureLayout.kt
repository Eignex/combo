package combo.bandit.glm

import combo.decisions.BoolContextHandle
import combo.decisions.CompiledDecisionSpace
import combo.decisions.IntContextHandle

/**
 * Linear-bandit-specific feature index over a [CompiledDecisionSpace]. Maps every
 * decision and context handle to a stable slot in the dense weight vector that the
 * linear models train against.
 *
 * Layout: `[bool decisions (numBoolVars), int decisions (numIntVars), bool contexts, int contexts]`
 *
 * Trees and other non-linear bandits don't go through this layer — they own their own
 * typed projection over the same [CompiledDecisionSpace].
 */
class LinearFeatureLayout internal constructor(
    val numBoolDecisions: Int,
    val numIntDecisions: Int,
    val boolContexts: List<BoolContextHandle>,
    val intContexts: List<IntContextHandle>,
) {
    val boolDecisionsStart: Int = 0
    val intDecisionsStart: Int = numBoolDecisions
    val boolContextsStart: Int = numBoolDecisions + numIntDecisions
    val intContextsStart: Int = boolContextsStart + boolContexts.size

    val featureSize: Int = intContextsStart + intContexts.size

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
        )
    }
}
