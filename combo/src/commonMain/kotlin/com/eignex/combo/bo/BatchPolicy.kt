package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context
import com.eignex.combo.shared.WitnessTermination
import com.eignex.klause.solver.Sample

/** How a sequential batch models outstanding outcomes. */
sealed interface FantasyPolicy {
    /** Exclude reservations without inventing temporary labels. */
    data object Exclusion : FantasyPolicy

    /** Refit temporary models using a fixed lie in original reward units. */
    data class ConstantLiar(
        /** Temporary response in original reward units. */
        val value: Double,
    ) : FantasyPolicy {
        init {
            require(value.isFinite())
        }
    }

    /** Refit temporary models using the base surrogate's predicted pending means. */
    data object PosteriorMean : FantasyPolicy
}

/** Per-round model/search callbacks. Temporary data must never be appended to the observation store. */
interface BatchSelector {
    /** Select one point conditioned on the supplied temporary observations. */
    fun select(temporary: List<Observation>, excluded: List<Sample>): CandidateSelection

    /** Optional joint search capability; implementations must not invent or record pending outcomes. */
    fun selectJoint(
        count: Int,
        acquisition: JointAcquisitionFactory,
        optimizer: JointAcquisitionOptimizer,
    ): ProposalBatch = throw UnsupportedOperationException("this selector does not support joint batches")

    /** Estimate an unevaluated point in original reward units for a temporary fantasy. */
    fun predict(sample: BanditSample, context: Context): Double
}

/** Batch construction independent of study reservations and numerical fitting. */
fun interface BatchPolicy {
    /** Construct a batch from one frozen snapshot; the study commits IDs only after this returns. */
    fun select(snapshot: StudySnapshot, count: Int, context: Context, selector: BatchSelector): ProposalBatch
}

/** Sequential selection with explicit pending exclusion and optional temporary full-history refits. */
class SequentialBatchPolicy(
    /** Temporary outcome strategy, never interpreted as actual feedback. */
    val fantasy: FantasyPolicy = FantasyPolicy.Exclusion,
) : BatchPolicy {
    override fun select(snapshot: StudySnapshot, count: Int, context: Context, selector: BatchSelector): ProposalBatch {
        require(count > 0)
        val temporary = mutableListOf<Observation>()
        val excluded = snapshot.pending.map { it.sample.sample }.toMutableList()
        fun fantasize(sample: BanditSample, ctx: Context) {
            val reward = when (val strategy = fantasy) {
                FantasyPolicy.Exclusion -> return
                is FantasyPolicy.ConstantLiar -> strategy.value
                FantasyPolicy.PosteriorMean -> selector.predict(sample, ctx)
            }
            temporary += Observation(sample, ctx, reward)
        }
        for (trial in snapshot.pending) fantasize(trial.sample, trial.context)
        val batch = mutableListOf<BanditSample>()
        repeat(count) {
            val selected = selector.select(temporary.toList(), excluded.toList())
            val point = selected.sample ?: return ProposalBatch(batch, selected.termination)
            batch += point
            excluded += point.sample
            fantasize(point, context)
        }
        return ProposalBatch(batch, WitnessTermination.CountReached)
    }
}

/**
 * Optimize a whole batch under one frozen joint posterior. Exploration and unseen-context warmup use
 * the selector's ordinary feasible proposals. Joint Gaussian moments are required after warmup;
 * bootstrap Thompson sampling is a separate incompatible suggestion strategy.
 *
 * @property acquisition Joint set utility, such as q-EI or q-NEI.
 * @property optimizer Feasible whole-set search.
 */
class JointBatchPolicy(val acquisition: JointAcquisitionFactory, val optimizer: JointAcquisitionOptimizer) :
    BatchPolicy {
    override fun select(snapshot: StudySnapshot, count: Int, context: Context, selector: BatchSelector): ProposalBatch {
        require(count > 0)
        return selector.selectJoint(count, acquisition, optimizer)
    }
}
