package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context

/** Identity of a reproducible external measurement, separate from model features and statistical weight. */
data class EvaluationKey(
    /** Dataset or problem instance; null for a single instance. */
    val instance: String? = null,
    /** External objective's random seed. */
    val seed: Int = 0,
    /** Optional fixed fidelity; a race compares keys at the same fidelity. */
    val budget: Double? = null,
) {
    init {
        require(budget == null || (budget > 0.0 && budget.isFinite()))
    }
}

/** External worker lifecycle, independent of the underlying configuration reservation. */
enum class EvaluationStatus {
    /** Issued to a worker. */
    Pending,

    /** Successful measurement. */
    Completed,

    /** Failed measurement without a fabricated response. */
    Failed,

    /** Released without an outcome. */
    Cancelled,
}

/** One independently identified worker request within a configuration race. */
data class EvaluationTask(
    /** Worker request identifier. */
    val id: Long,
    /** Underlying BO configuration trial identifier. */
    val configurationId: Long,
    /** Feasible original assignment. */
    val sample: BanditSample,
    /** Fixed hard context of this race. */
    val context: Context,
    /** Paired measurement identity. */
    val key: EvaluationKey,
    /** Captured lifecycle state. */
    val status: EvaluationStatus = EvaluationStatus.Pending,
)

/** Closed worker request with observed response, cost, and optional known measurement variance. */
data class EvaluationRecord(
    /** Closed task snapshot. */
    val task: EvaluationTask,
    /** Finite measured response, present only for successful completion. */
    val reward: Double? = null,
    /** Nonnegative external evaluation cost, including unsuccessful work. */
    val cost: Double = 1.0,
    /** Optional nonnegative worker duration. */
    val seconds: Double? = null,
    /** Known variance in original squared response units. */
    val noiseVariance: Double? = null,
    /** Failure or cancellation explanation. */
    val reason: String? = null,
) {
    init {
        require(task.status != EvaluationStatus.Pending)
        require(if (task.status == EvaluationStatus.Completed) reward != null && reward.isFinite() else reward == null)
        require(cost >= 0.0 && cost.isFinite())
        require(seconds == null || (seconds >= 0.0 && seconds.isFinite()))
        require(noiseVariance == null || (noiseVariance >= 0.0 && noiseVariance.isFinite()))
    }
}

/** Immutable external measurement snapshot. */
data class EvaluationHistory(
    /** Append revision. */
    val revision: Long,
    /** Closed measurements in completion order. */
    val records: List<EvaluationRecord>,
)

/** Append-only external evaluation storage; failed appends leave both data and revision unchanged. */
interface EvaluationStore {
    /** Monotone append revision. */
    val revision: Long

    /** Atomically append one closed task, rejecting duplicate task identifiers. */
    fun append(record: EvaluationRecord)

    /** Capture records independent of later appends. */
    fun snapshot(): EvaluationHistory
}

/** Single-threaded full-history evaluation store. */
class InMemoryEvaluationStore : EvaluationStore {
    private val records = mutableListOf<EvaluationRecord>()
    private val ids = mutableSetOf<Long>()
    override var revision = 0L
        private set
    override fun append(record: EvaluationRecord) {
        require(record.task.id !in ids) { "evaluation already recorded" }
        records += record
        ids += record.task.id
        revision++
    }
    override fun snapshot() = EvaluationHistory(revision, records.toList())
}
