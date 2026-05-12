package combo.bandit.ga

import combo.bandit.univariate.*
import combo.ga.TournamentElimination
import combo.math.RunningVariance
import combo.math.nextNormal
import combo.model.TestModels.MODEL1
import combo.sat.Problem
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class EliminationChainTest {
    @Test
    fun noEliminationOnEmpty() {
        val candidates = createCandidates(MODEL1.problem, 20, UCB1())
        val s = EliminationChain(SignificanceTestElimination(0.999f), TournamentElimination(10))
        val select = s.select(candidates, Random)
        assertTrue(select < 0)
    }

    @Test
    fun guaranteedElimination() {
        val p = Problem(20, emptyArray())
        val candidates = createCandidates(p, 10, UCB1Normal(), minSamples = 0f)
        val chain = EliminationChain(SignificanceTestElimination(), TournamentElimination(3))
        val e = chain.select(candidates, Random)
        assertTrue(e >= 0)
    }

    @Test
    fun eliminate() {
        val candidates = createCandidates(MODEL1.problem, 20, EpsilonDecreasing(), minSamples = 4.0f)
        val s = EliminationChain(SignificanceTestElimination(0.5f), SmallestCountElimination())

        val rng = Random
        do {
            val i = rng.nextInt(10)
            val value = i.toFloat() + rng.nextNormal()
            candidates.estimators[candidates.instances[i]]!!.accept(value)
        } while (s.select(candidates, rng) < 0)

        val eliminated = s.select(candidates, rng)
        val elimE = candidates.estimators[candidates.instances[eliminated]]!!
        assertEquals(4.0f, elimE.nbrWeightedSamples)
    }
}

class SignificanceTestEliminationTest {

    @Test
    fun noEliminationOnEmpty() {
        val candidates = createCandidates(MODEL1.problem, 10, ThompsonSampling(NormalPosterior))
        val s = SignificanceTestElimination(0.999f)
        assertTrue(s.select(candidates, Random) < 0)
    }

    @Test
    fun eliminateMinimization() {
        val candidates = createCandidates(MODEL1.problem, 10, ThompsonSampling(NormalPosterior), maximize = false, minSamples = 4.0f)
        val s = SignificanceTestElimination(0.05f)

        val rng = Random
        do {
            val i = rng.nextInt(10)
            val value = i.toFloat() + rng.nextNormal()
            candidates.estimators[candidates.instances[i]]!!.accept(value)
        } while (s.select(candidates, rng) < 0)

        val eliminated = s.select(candidates, rng)
        val elimE = candidates.estimators[candidates.instances[eliminated]]!!
        assertTrue(elimE.nbrWeightedSamples >= candidates.minSamples)

        val best = candidates.estimators.values.minByOrNull {
            if (it.nbrWeightedSamples < candidates.minSamples) Float.POSITIVE_INFINITY
            else it.mean + s.z * it.standardDeviation / sqrt(it.nbrWeightedSamples)
        }!!
        val bestUpper = best.mean + s.z * best.standardDeviation / sqrt(best.nbrWeightedSamples)
        val elimELower = elimE.mean - s.z * elimE.standardDeviation / sqrt(elimE.nbrWeightedSamples)
        assertTrue(bestUpper < elimELower)
    }

    @Test
    fun eliminateMaximization() {
        val candidates = createCandidates(MODEL1.problem, 10, ThompsonSampling(NormalPosterior), maximize = true, minSamples = 4.0f)
        val s = SignificanceTestElimination(0.05f)

        val rng = Random
        do {
            val i = rng.nextInt(10)
            val value = i.toFloat() + rng.nextNormal()
            candidates.estimators[candidates.instances[i]]!!.accept(value)
        } while (s.select(candidates, rng) < 0)

        val eliminated = s.select(candidates, rng)
        val elimE = candidates.estimators[candidates.instances[eliminated]]!!
        assertTrue(elimE.nbrWeightedSamples >= candidates.minSamples)

        val best = candidates.estimators.values.maxByOrNull {
            if (it.nbrWeightedSamples < candidates.minSamples) Float.NEGATIVE_INFINITY
            else it.mean - s.z * it.standardDeviation / sqrt(it.nbrWeightedSamples)
        }!!
        val bestLower = best.mean - s.z * best.standardDeviation / sqrt(best.nbrWeightedSamples)
        val elimEUpper = elimE.mean + s.z * elimE.standardDeviation / sqrt(elimE.nbrWeightedSamples)
        assertTrue(bestLower > elimEUpper)
    }
}

class SmallestCountEliminiationTest {

    @Test
    fun noEliminationOnEmpty() {
        val candidates = createCandidates(MODEL1.problem, 10, ThompsonSampling(NormalPosterior))
        val s = SmallestCountElimination()
        val rng = Random(0)
        assertTrue(s.select(candidates, rng) < 0)
    }

    @Test
    fun eliminate() {
        val candidates = createCandidates(MODEL1.problem, 20, ThompsonSampling(NormalPosterior, RunningVariance()), minSamples = 4.0f)
        val s = SmallestCountElimination()

        val rng = Random
        do {
            val i = rng.nextInt(10)
            val value = i.toFloat() + rng.nextNormal()
            candidates.estimators[candidates.instances[i]]!!.accept(value)
        } while (s.select(candidates, rng) < 0)

        val eliminated = s.select(candidates, rng)
        val elimE = candidates.estimators[candidates.instances[eliminated]]!!
        assertEquals(4.0f, elimE.nbrWeightedSamples)
    }
}
