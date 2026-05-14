package combo.bandit.dt

import combo.decisions.BanditSample
import combo.decisions.CompiledDecisionSpace
import combo.decisions.Context
import combo.decisions.FeatureEncoder

/**
 * Tree bandit's view of a [CompiledDecisionSpace]. Projects a klause [Sample] (plus
 * [Context] for parity with the [FeatureEncoder] contract — the row uses the sample's
 * pinned-context values directly) into a [TreeRow] the splits can query by handle.
 *
 * Unlike [combo.bandit.glm.LinearFeatureProjection], trees consume typed handle access
 * rather than a dense vector. There is no `toObjective` here: a learned tree's
 * predictions don't reduce to a linear klause objective, so choose-time relies on the
 * bandit sampling candidates from a klause [com.eignex.klause.solver.Sampler] and
 * scoring them through the tree.
 *
 * Slice 1: encoder skeleton only. Slices 2–4 will add split-finding metrics, the tree
 * node types, and the [DecisionTreeBandit] / `RandomForestBandit` orchestration.
 */
class TreeFeatureProjection(override val space: CompiledDecisionSpace) : FeatureEncoder<TreeRow> {

    /**
     * Number of typed "columns" a tree may split on: one per klause variable, plus one
     * per optional variable's `isPresent` indicator (once that slice lands).
     *
     * This count is descriptive only — trees don't index columns by position the way
     * the linear projection does. It exists to satisfy [FeatureEncoder.featureSize]
     * and to drive `mtry`-style random subspace selection in forest training.
     */
    override val featureSize: Int =
        space.compiled.problem.numBoolVars + space.compiled.problem.numIntVars

    override fun encode(sample: BanditSample, context: Context): TreeRow =
        SampleTreeRow(space, sample)
}
