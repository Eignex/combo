package combo.bandit.ga

import combo.bandit.univariate.BanditPolicy
import combo.bandit.univariate.NormalPosterior
import combo.bandit.univariate.ThompsonSampling
import combo.math.RunningVariance
import combo.model.TestModels.MODEL1
import combo.model.TestModels.MODEL3
import combo.sat.BitArray
import combo.sat.Instance
import combo.sat.Problem
import combo.sat.optimizers.LocalSearch
import kotlin.test.Test
import kotlin.test.assertEquals

fun createCandidates(problem: Problem, n: Int, bp: BanditPolicy,
                     minSamples: Float = 5.0f, maximize: Boolean = true): BanditCandidates {
    val instances: Array<Instance> = LocalSearch(problem).asSequence().take(n).toList().toTypedArray()
    return BanditCandidates(instances, minSamples, maximize, bp)
}

class BanditCandidatesTest {

    @Test
    fun createOne() {
        val candidates = createCandidates(MODEL1.problem, 1, ThompsonSampling(NormalPosterior, RunningVariance()))
        assertEquals(0, candidates.oldestCandidate)
        val e = candidates.estimators[candidates.instances[0]]!!
        assertEquals(e.mean, candidates.worstScore, 0.0f)
        assertEquals(e.mean, candidates.bestScore, 0.0f)
    }

    @Test
    fun minMaxScoreAfterUpdate() {
        val candidates1 = createCandidates(MODEL3.problem, 20, ThompsonSampling(NormalPosterior, RunningVariance()), maximize = true)
        candidates1.update(candidates1.instances[2], 1.0f, 1.0f)
        assertEquals(-1.0f, candidates1.bestScore, 0.0f)
        assertEquals(0.0f, candidates1.worstScore, 0.0f)

        val candidates2 = createCandidates(MODEL3.problem, 20, ThompsonSampling(NormalPosterior, RunningVariance()), maximize = false)
        candidates2.update(candidates2.instances[2], 1.0f, 1.0f)
        assertEquals(0.0f, candidates2.bestScore, 0.0f)
        assertEquals(1.0f, candidates2.worstScore, 0.0f)
    }

    @Test
    fun candidatesWithAge() {
        val problem = Problem(1, emptyArray())
        val candidates = createCandidates(problem, 5, ThompsonSampling(NormalPosterior))
        assertEquals(0, candidates.oldestCandidate)
        candidates.update(candidates.instances[0], 1.0f, 1.0f)
        candidates.replaceCandidate(0, BitArray(1))
        assertEquals(1, candidates.oldestCandidate)
    }
}
