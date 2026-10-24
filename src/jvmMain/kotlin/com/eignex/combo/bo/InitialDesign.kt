package com.eignex.combo.bo

import com.eignex.combo.decisions.Context
import com.eignex.klause.solver.Sample
import kotlin.random.Random

/** Frozen inputs for a single initial-design slot within a batch. */
class InitialDesignRequest(
    /** State at the beginning of the round. */
    val snapshot: StudySnapshot,
    /** Hard caller context. */
    val context: Context,
    /** Pending or completed configurations unavailable for this slot. */
    val excluded: List<Sample>,
    /** Number of earlier selections in this batch. */
    val offset: Int,
)

/** Decide whether a round uses random initial/interleaved exploration. */
fun interface InitialDesign {
    /** Decide once per frozen batch, before fitting a surrogate. */
    fun explore(snapshot: StudySnapshot, random: Random): Boolean

    /** Optional structured initial proposal; null delegates to independent feasible random exploration. */
    fun sample(request: InitialDesignRequest, random: Random): CandidateSelection? = null
}

/** Initial observations followed by randomly interleaved exploration. */
class RandomInitialDesign(
    /** Number of observations before model-driven search. */
    val observations: Int = 5,
    /** Exploration probability after initial design. */
    val probability: Double = 0.1,
) : InitialDesign {
    init {
        require(observations >= 0 && probability in 0.0..1.0)
    }
    override fun explore(snapshot: StudySnapshot, random: Random): Boolean =
        snapshot.history.observations.isEmpty() || snapshot.history.observations.size < observations ||
            random.nextDouble() < probability
}
