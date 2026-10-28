package com.eignex.combo.shared

import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.klause.solver.Sample
import com.eignex.klause.solver.checkExactRealSample
import com.eignex.klause.solver.evidence.SourceClaim
import com.eignex.klause.solver.evidence.SourceDerivation
import com.eignex.klause.solver.evidence.SourceEvidence
import com.eignex.klause.solver.evidence.SourceEvidenceCheck
import com.eignex.klause.solver.evidence.SourceEvidenceVerifier
import com.eignex.klause.solver.evidence.SourceRevision
import com.eignex.klause.solver.evidence.SourceScope
import com.eignex.klause.solver.evidence.sourceProofContext
import com.eignex.klause.util.Cancellation

internal class WitnessValidator(private val space: CompiledDecisionSpace) {
    private val scope = SourceScope(SourceRevision("combo/witness", "klause-ir-exact-v1"))
    private val verifier by lazy {
        space.coordinates.problem.sourceProofContext(scope, cancellation = budget())?.let(::SourceEvidenceVerifier)
    }
    private val checked = linkedMapOf<Sample, Boolean>()

    fun valid(sample: Sample): Boolean {
        checked[sample]?.let { return it }
        if (space.coordinates.schema == null && (sample.symbolicReals != null || sample.algebraicReals != null)) {
            return space.problem.checkExactRealSample(sample, budget()) == true
        }
        val valid = verifier?.verify(
            SourceEvidence(scope, SourceClaim.Witness(sample), SourceDerivation.SourceWitness),
            budget(),
        ) == SourceEvidenceCheck.Checked
        if (checked.size >= 4096) checked.remove(checked.keys.first())
        checked[sample] = valid
        return valid
    }

    private fun budget(): Cancellation {
        var polls = 0
        return Cancellation { ++polls > 100_000 }
    }
}
