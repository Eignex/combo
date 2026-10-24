package com.eignex.combo.bo

import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.shared.WitnessValidator
import com.eignex.combo.shared.assignmentKey
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.solver.Sample

internal class HistoryCandidates(private val space: CompiledDecisionSpace) {
    private val validator = WitnessValidator(space)

    fun valid(sample: Sample): Boolean = validator.valid(sample)

    fun best(
        history: ObservationHistory,
        assumptions: Assumptions,
        excluded: List<Sample>,
        score: (Sample) -> Double,
    ): Sample? {
        val keys = excluded.map { it.assignmentKey() }.toSet()
        return history.observations.asSequence().map { it.sample.sample }.distinctBy { it.assignmentKey() }
            .filter { it.assignmentKey() !in keys && space.matches(it, assumptions) && valid(it) }.maxByOrNull(score)
    }
}
