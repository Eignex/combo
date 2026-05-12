package combo.decisions

import com.eignex.klause.compile.CompiledProblem

/**
 * Maps every handle declared on a [DecisionSpace] to a stable index in the dense feature
 * vector consumed by linear bandits.
 *
 * Slice 1 layout (no optionals, no interactions, no nominals expanded):
 *   `[bool decisions (numBoolVars), int decisions (numIntVars), bool contexts, int contexts]`
 */
class FeatureLayout internal constructor(
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
        internal fun from(
            compiled: CompiledProblem,
            contextBools: List<BoolContextHandle>,
            contextInts: List<IntContextHandle>,
        ): FeatureLayout = FeatureLayout(
            numBoolDecisions = compiled.problem.numBoolVars,
            numIntDecisions = compiled.problem.numIntVars,
            boolContexts = contextBools.toList(),
            intContexts = contextInts.toList(),
        )
    }
}
