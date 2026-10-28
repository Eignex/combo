package com.eignex.combo.shared

import com.eignex.klause.arithmetic.real.exactRealValue
import com.eignex.klause.formats.json.ExactRealJson
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.solver.Sample
import kotlin.random.Random

/** Why a bounded witness request stopped; a short batch alone never establishes infeasibility. */
enum class WitnessTermination {
    /** The requested number of distinct witnesses was returned. */
    CountReached,

    /** The attempt allowance ended, possibly after finding a partial batch. */
    BudgetExhausted,

    /** The solver proved the original hard scope infeasible. */
    Infeasible,
}

/**
 * Batch feasibility request in original Klause coordinates.
 *
 * @property count Maximum number of distinct witnesses to return.
 * @property assumptions Hard scope respected by every returned witness.
 * @property anchors External candidate seeds, independently checked before reuse.
 * @property exclude Assignments reserved elsewhere; equality ignores proof metadata.
 * @property maxAttempts Maximum proposal and fallback calls combined. Engine budgets apply per call.
 * @property context Complete context for native real conditioning alongside discrete assumptions.
 */
data class WitnessRequest(
    val count: Int,
    val assumptions: Assumptions = Assumptions.None,
    val anchors: List<Sample> = emptyList(),
    val exclude: List<Sample> = emptyList(),
    val maxAttempts: Int = 128,
    /** Real and discrete context retained for conditioned native search. */
    val context: com.eignex.combo.decisions.Context = com.eignex.combo.decisions.Context.Empty,
) {
    init {
        require(count > 0 && maxAttempts >= 0) { "invalid batch count or attempt allowance" }
    }
}

/** A partial or complete batch of distinct feasible assignments and its termination reason. */
class WitnessBatch(
    witnesses: List<Sample>,
    /** Reason candidate generation stopped. */
    val termination: WitnessTermination,
) {
    /** Stable list of feasible assignments, independent of later search calls. */
    val witnesses: List<Sample> = witnesses.toList()
}

/** Batch candidate generation shared by streaming and history-based learners. Calls are single-threaded. */
fun interface WitnessSearch {
    /** Request feasible candidates; excluded or duplicate points do not count toward the requested size. */
    fun witnesses(request: WitnessRequest, random: Random): WitnessBatch
}

internal data class AssignmentKey(val bools: List<Boolean>, val ints: List<String>, val reals: List<Any>)

internal fun Sample.assignmentKey(): AssignmentKey = AssignmentKey(
    bools.toList(),
    List(numIntVars) { exactIntValue(it).toString() },
    List(numRealVars) { ExactRealJson.encodeToJsonElement(exactRealValue(it)) },
)
