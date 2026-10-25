package com.eignex.combo.bandit.dt

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.FeatureEncoder

/**
 * Tree bandit's view of a [CompiledDecisionSpace]. Projects a klause [com.eignex.klause.solver.Sample] (plus
 * [Context] for parity with the [FeatureEncoder] contract — the row uses the sample's
 * pinned-context values directly) into a [TreeRow] the splits can query by handle.
 *
 * [TreeVectorLayout] supplies Kumulant with a numeric view of these typed values and
 * retains the original predicates for constraint-aware descent. Learned tree scores
 * drive feasible-candidate ranking and branch selection through Klause.
 */
class TreeFeatureProjection(override val space: CompiledDecisionSpace) : FeatureEncoder<TreeRow> {

    /**
     * Number of solver coordinates exposed to trees, including native continuous
     * variables and the Boolean presence gates registered by Klause.
     *
     * This is the solver-coordinate count. [TreeVectorLayout.featureSize] describes
     * the model's numeric view, including derived predicate features; Kumulant draws
     * its per-leaf random subspaces from the split-candidate pool.
     */
    override val featureSize: Int =
        space.compiled.problem.numBoolVars + space.compiled.problem.numIntVars + space.compiled.problem.numRealVars

    override fun encode(sample: BanditSample, context: Context): TreeRow = SampleTreeRow(space, sample)
}
