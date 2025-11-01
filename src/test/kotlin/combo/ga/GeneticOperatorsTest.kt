package combo.ga

import combo.math.permutation
import combo.expressions.Tautology
import combo.model.TestModels
import combo.model.TestModels.MODEL1
import combo.model.TestModels.MODEL3
import combo.model.TestModels.MODEL5
import combo.sat.*
import combo.util.IntHashSet
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

data class ExpandedCandidates(val candidates: ValidatorCandidates,
                              val instances: Array<Validator>,
                              val scores: FloatArray)

fun createCandidates(problem: Problem, nbrStates: Int, rng: Random): ExpandedCandidates {
    val instanceBuilder = BitArrayFactory
    val instances = Array(nbrStates) {
        val instance = instanceBuilder.create(problem.nbrValues)
        WordRandomSet().initialize(instance, Tautology, rng, null)
        Validator(problem, instance, Tautology)
    }
    val scores = FloatArray(nbrStates) { instances[it].totalUnsatisfied.toFloat() }
    val candidates = ValidatorCandidates(instances, IntArray(nbrStates), scores)
    return ExpandedCandidates(candidates, instances, scores)
}

class ValidatorCandidatesTest {

    @Test
    fun createOne() {
        val (candidates, _, scores) = createCandidates(MODEL1.problem, 1, Random)
        assertEquals(0, candidates.oldestCandidate)
        assertEquals(scores[0], candidates.worstScore)
        assertEquals(scores[0], candidates.bestScore)
    }

    @Test
    fun minMaxScore() {
        val (candidates, instances, _) = createCandidates(MODEL3.problem, 20, Random)
        val min = instances.map { it.totalUnsatisfied.toFloat() }.minOrNull()!!
        val max = instances.map { it.totalUnsatisfied.toFloat() }.maxOrNull()!!
        assertEquals(min, candidates.bestScore)
        assertEquals(max, candidates.worstScore)
    }

    @Test
    fun candidatesWithAge() {
        val origins = intArrayOf(1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6)
        val scores = FloatArray(origins.size) { it.toFloat() }
        val problem = Problem(1, emptyArray())
        val validators = Array(origins.size) {
            val instance = BitArray(1)
            Validator(problem, instance, Tautology)
        }
        val candidates = ValidatorCandidates(validators, origins, scores)
        assertEquals(6, candidates.oldestCandidate)
        assertEquals(1, candidates.oldestOrigin)
    }

    @Test
    fun oldestAfterUpdate() {
        val origins = intArrayOf(1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6)
        val scores = FloatArray(origins.size) { it.toFloat() }
        val problem = Problem(10, emptyArray())
        val validators = Array(origins.size) {
            val instance = BitArray(10)
            WordRandomSet().initialize(instance, Tautology, Random, null)
            Validator(problem, instance, Tautology)
        }
        val candidates = ValidatorCandidates(validators, origins, scores)
        assertEquals(6, candidates.oldestCandidate)
        assertEquals(1, candidates.oldestOrigin)
        candidates.update(6, 10, 0.0f)
        assertEquals(0, candidates.oldestCandidate)
        assertEquals(1, candidates.oldestOrigin)
    }

    @Test
    fun changeToYoungerOrigin() {
        // This tests the similar functionality in GeneticAlgorithmOptimizer that changes the origin to something older
        val origins = IntArray(20) { 10 + it }
        val scores = FloatArray(20) { it.toFloat() }
        val problem = Problem(10, emptyArray())
        val validators = Array(origins.size) {
            val instance = BitArray(10)
            WordRandomSet().initialize(instance, Tautology, Random, null)
            Validator(problem, instance, Tautology)
        }

        val candidates = ValidatorCandidates(validators, origins, scores)
        val keep = IntHashSet(nullValue = -1).apply { addAll(0 until 5) }
        for (i in 0 until candidates.nbrCandidates) {
            if (i in keep) {
                candidates.update(i, 1, -1.0f)
            } else {
                candidates.update(i, 0, 1.0f)
            }
        }
        assertTrue(candidates.oldestCandidate >= 5)
        assertEquals(0, candidates.oldestOrigin)
    }
}

abstract class RecombinationOperatorTest {

    abstract fun crossoverOperator(): RecombinationOperator<ValidatorCandidates>

    @Test
    fun crossoverSelf() {
        val p = MODEL5.problem
        val (candidates, trackers) = createCandidates(p, 10, Random)
        val crossoverOperator = crossoverOperator()
        val instance1 = candidates.instances[0].copy()
        crossoverOperator.combine(0, 0, 0, candidates, Random)
        val instance2 = trackers[0]
        assertEquals(instance1, instance2.instance)
    }

    @Test
    fun testDifference() {
        for (p in TestModels.SAT_PROBLEMS + TestModels.UNSAT_PROBLEMS + TestModels.LARGE_PROBLEMS) {
            val rng = Random
            val popSize = 10
            val (candidates, _) = createCandidates(p, popSize, rng)
            val perm = permutation(popSize, rng)
            val p1 = perm.encode(0)
            val p2 = perm.encode(1)
            val c = perm.encode(2)
            val instance1 = candidates.instances[p1].copy()
            val instance2 = candidates.instances[p2].copy()
            val crossoverOperator = crossoverOperator()
            crossoverOperator.combine(p1, p2, c, candidates, rng)
            for (i in 0 until instance1.size) {
                assertEquals(instance1.isSet(i), candidates.instances[p1].isSet(i))
                assertEquals(instance2.isSet(i), candidates.instances[p2].isSet(i))

                if (instance1.isSet(i) == instance2.isSet(i)) assertEquals(instance1.isSet(i), candidates.instances[c].isSet(i), "$i")
                else assertTrue(instance1.isSet(i) == candidates.instances[c].isSet(i) || instance2.isSet(i) == candidates.instances[c].isSet(i))
            }
        }
    }
}

class UniformRecombinationTest : RecombinationOperatorTest() {
    override fun crossoverOperator() = UniformRecombination()
}

class KPointRecombinationTest : RecombinationOperatorTest() {
    override fun crossoverOperator() = KPointRecombination(1)
}

class KPoint2CrossoverTest : RecombinationOperatorTest() {
    override fun crossoverOperator() = KPointRecombination(2)
}

abstract class SelectionOperatorTest {

    abstract fun selectionOperator(popSize: Int): SelectionOperator<Candidates>

    @Test
    fun totalSpread() {
        for (p in TestModels.SAT_PROBLEMS + TestModels.UNSAT_PROBLEMS + TestModels.LARGE_PROBLEMS) {
            val rng = Random(1)
            val n = 10
            val (candidates, _) = createCandidates(p, n, rng)
            val selectionOperator = selectionOperator(n)
            val selected = generateSequence { selectionOperator.select(candidates, rng) }.take(500).toSet()
            // Uniquely worst genome can never be selected by tournament with size=2
            assertTrue(selected.size + 1 >= n)
        }
    }
}

class UniformSelectionTest : SelectionOperatorTest() {
    override fun selectionOperator(popSize: Int) = UniformSelection()
}

class StochasticAcceptanceSelectionTest : SelectionOperatorTest() {
    override fun selectionOperator(popSize: Int) = StochasticAcceptanceSelection()
}

class TournamentSelectionTest : SelectionOperatorTest() {
    override fun selectionOperator(popSize: Int) = TournamentSelection(2)
}

class OldestEliminationTest {
    @Test
    fun selectOldest() {
        val (candidates, _) = createCandidates(TestModels.SAT_PROBLEMS[0], 10, Random)
        for (i in 0..8) {
            candidates.update(i, (i + 1).toLong(), candidates.scores[i])
        }
        assertEquals(9, OldestElimination().select(candidates, Random))
    }
}

abstract class MutationRateTest {

    abstract fun rate(nbrVariables: Int): MutationRate

    @Test
    fun mutationRate() {
        for (n in 1..20) {
            val mutator = rate(n)
            val rate = mutator.rate(n, Random)
            assertTrue(rate > 0 && rate <= 1, "$n")
        }
    }
}

class FastGAMutationTest : MutationRateTest() {
    override fun rate(nbrVariables: Int) = FastGAMutation(nbrVariables, 1.5f)
}

class FixedRateMutationTest : MutationRateTest() {
    override fun rate(nbrVariables: Int) = FixedRateMutation()
}

class FixedRate2MutationTest : MutationRateTest() {
    override fun rate(nbrVariables: Int) = FixedRateMutation(2)
}

abstract class MutationOperatorTest {

    abstract fun mutationOperator(problem: Problem): MutationOperator<ValidatorCandidates>

    @Test
    fun mutateCanChange() {
        for (n in 1..20) {
            val problem = Problem(n, emptyArray())
            val rng = Random(n)
            val (candidates, _, _) = createCandidates(problem, 1, rng)
            val mutator = mutationOperator(problem)
            val preMutated = candidates.instances[0].instance.copy()
            val postMutated = candidates.instances[0]
            var itr = 0
            do {
                mutator.mutate(0, candidates, rng)
                assertTrue(itr++ <= 50, "Max iterations reached")
            } while (preMutated == postMutated)
        }
    }

    @Test
    fun mutateCanChangeWithProblem() {
        val problem = MODEL1.problem
        val rng = Random(problem.nbrConstraints)
        val (candidates, _, _) = createCandidates(problem, 1, rng)
        val mutator = mutationOperator(problem)
        val preMutated = candidates.instances[0].instance.copy()
        val postMutated = candidates.instances[0]
        var itr = 0
        do {
            mutator.mutate(0, candidates, rng)
            assertTrue(itr++ <= 50, "Max iterations reached")
        } while (preMutated == postMutated)
    }
}

class RateMutationOperatorTest : MutationOperatorTest() {
    override fun mutationOperator(problem: Problem) = RateMutationOperator(FixedRateMutation(1))
}

class FixedMutationOperatorTest : MutationOperatorTest() {
    override fun mutationOperator(problem: Problem) = FixedMutation(1)
}

class PropagatingMutatorTest : MutationOperatorTest() {
    override fun mutationOperator(problem: Problem) = PropagatingMutator(FixedRateMutation(1), TransitiveImplications(problem))
}
