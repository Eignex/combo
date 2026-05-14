package combo.bandit.dt

import com.eignex.klause.solver.Sample
import kotlin.random.Random

/**
 * Shared choose-time loop: ask [proposeSample] for [budget] feasible candidates and
 * return the one with the highest [scoreSample] value. Returns `null` if no proposer
 * call yielded a sample within the budget.
 *
 * Both [DecisionTreeBandit] and [RandomForestBandit] use this; they differ only in
 * how [scoreSample] interprets a candidate (walk one tree vs. aggregate across trees).
 */
internal inline fun pickBestSample(
    rng: Random,
    budget: Int,
    proposeSample: (Random) -> Sample?,
    scoreSample: (Sample) -> Double,
): Sample? {
    var best: Sample? = null
    var bestScore = Double.NEGATIVE_INFINITY
    var i = 0
    while (i < budget) {
        val s = proposeSample(rng)
        if (s != null) {
            val v = scoreSample(s)
            if (v > bestScore) {
                bestScore = v
                best = s
            }
        }
        i++
    }
    return best
}
