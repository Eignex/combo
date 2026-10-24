package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context
import com.eignex.combo.shared.WitnessTermination

/** Evaluation lifecycle; only successful completion contributes an observation to the model. */
enum class TrialStatus {
    /** Reserved and awaiting an evaluation outcome. */
    Pending,

    /** Successfully evaluated and recorded in the observation store. */
    Completed,

    /** Evaluation failed; its assignment is available for future suggestions. */
    Failed,

    /** Reservation was released without an evaluation. */
    Cancelled,
}

/**
 * Immutable snapshot of an optimizer-owned trial. IDs are scoped to one optimizer and its exported state.
 *
 * @property id Stable reservation identifier.
 * @property sample Feasible assignment in original coordinates.
 * @property context Hard context captured when suggesting the trial.
 * @property status Evaluation lifecycle state.
 * @property reward Observed outcome, present only after successful completion.
 * @property weight Observation weight, present only after successful completion.
 * @property reason Optional failure or cancellation explanation.
 */
@ConsistentCopyVisibility
data class Trial internal constructor(
    val id: Long,
    val sample: BanditSample,
    val context: Context,
    val status: TrialStatus = TrialStatus.Pending,
    val reward: Double? = null,
    val weight: Double? = null,
    val reason: String? = null,
)

/**
 * A possibly partial group of reserved evaluations. Iteration and indexing return trial snapshots.
 *
 * @param trials Reservations captured by this batch.
 * @property termination CountReached if all requested reservations were made; otherwise why search stopped.
 */
class TrialBatch(trials: List<Trial>, val termination: WitnessTermination) : Iterable<Trial> {
    /** Stable ordered snapshots; later completion does not modify this batch. */
    val trials: List<Trial> = trials.toList()

    /** Number of reservations made, possibly smaller than requested. */
    val size: Int get() = trials.size

    /** Return the trial snapshot at the given batch position. */
    operator fun get(index: Int): Trial = trials[index]
    override fun iterator(): Iterator<Trial> = trials.iterator()
}

/** Single-threaded asynchronous evaluation interface for history-based optimization. */
interface TrialOptimizer {
    /** Current snapshots of all trials issued by this optimizer, in creation order. */
    val trials: List<Trial>

    /** Reservations awaiting an outcome; they never count as observed rewards. */
    val pendingTrials: List<Trial> get() = trials.filter { it.status == TrialStatus.Pending }

    /** Reserve up to count distinct feasible assignments, excluding all outstanding reservations. */
    fun suggest(count: Int = 1, context: Context = Context.Empty): TrialBatch

    /** Record one pending trial exactly once; completion order need not match suggestion order. */
    fun complete(id: Long, reward: Double, weight: Double = 1.0)

    /** Close a pending trial without adding a training observation. */
    fun fail(id: Long, reason: String)

    /** Release a pending reservation without adding a training observation. */
    fun cancel(id: Long, reason: String? = null)
}
