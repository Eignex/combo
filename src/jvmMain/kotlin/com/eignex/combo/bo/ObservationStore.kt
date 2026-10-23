package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context

/**
 * One evaluated configuration, preserving context and its statistical weight.
 *
 * @property sample Evaluated solver assignment.
 * @property context Context supplied when evaluating the assignment.
 * @property reward Finite observed response.
 * @property weight Positive finite observation weight.
 */
data class Observation(val sample: BanditSample, val context: Context, val reward: Double, val weight: Double = 1.0) {
    init {
        require(reward.isFinite()) { "reward must be finite" }
        require(weight > 0.0 && weight.isFinite()) { "observation weight must be positive and finite" }
    }
}

/**
 * Consistent snapshot of an observation store, in insertion order.
 *
 * @property revision Store revision at capture time.
 * @property observations Observations independent of later store appends.
 */
data class ObservationHistory(val revision: Long, val observations: List<Observation>)

/**
 * Append-only observation history for one decision-space schema. Implementations may
 * keep observations in memory or persist them externally. A successful append advances
 * [revision]; [snapshot] captures that revision and its observations consistently.
 */
interface ObservationStore {
    /** Monotonically increasing revision used to invalidate fitted models. */
    val revision: Long

    /** Append one observation without combining or discarding earlier evaluations. */
    fun append(observation: Observation)

    /** Capture a stable history; subsequent appends must not modify the returned rows. */
    fun snapshot(): ObservationHistory
}

/** Single-threaded in-memory store retaining the entire observation history. */
class InMemoryObservationStore : ObservationStore {
    private val observations = mutableListOf<Observation>()

    override var revision: Long = 0L
        private set

    override fun append(observation: Observation) {
        observations += observation
        revision++
    }

    override fun snapshot(): ObservationHistory = ObservationHistory(revision, observations.toList())
}
