package com.eignex.combo.bo

import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.solver.Sample
import com.eignex.klause.solver.evidence.SourceClaim
import com.eignex.klause.solver.evidence.SourceDerivation
import com.eignex.klause.solver.evidence.SourceEvidence
import com.eignex.klause.solver.evidence.SourceEvidenceCheck
import com.eignex.klause.solver.evidence.SourceEvidenceVerifier
import com.eignex.klause.solver.evidence.SourceRevision
import com.eignex.klause.solver.evidence.SourceScope
import com.eignex.klause.solver.evidence.sourceProofContext
import com.eignex.klause.util.Cancellation

internal class HistoryCandidates(private val space: CompiledDecisionSpace) {
    private val scope = SourceScope(SourceRevision("combo/history", "klause-ir-exact-v1"))
    private val verifier by lazy {
        space.compiled.problem.sourceProofContext(scope, cancellation = budget())?.let(::SourceEvidenceVerifier)
    }
    private val checked = mutableMapOf<Sample, Boolean>()

    fun valid(sample: Sample): Boolean = checked.getOrPut(sample) {
        verifier?.verify(
            SourceEvidence(scope, SourceClaim.Witness(sample), SourceDerivation.SourceWitness),
            budget(),
        ) == SourceEvidenceCheck.Checked
    }

    fun best(history: ObservationHistory, assumptions: Assumptions, score: (Sample) -> Double): Sample? =
        history.observations.asSequence().map { it.sample.sample }.distinct()
            .filter { space.matches(it, assumptions) && valid(it) }.maxByOrNull(score)

    private fun budget(): Cancellation {
        var polls = 0
        return Cancellation { ++polls > 100_000 }
    }
}
