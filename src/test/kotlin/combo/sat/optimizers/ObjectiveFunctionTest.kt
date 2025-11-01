package combo.sat.optimizers

import combo.math.FloatMatrix
import combo.math.FloatVector
import combo.math.VectorView
import combo.math.nextNormal
import combo.expressions.Tautology
import combo.model.TestModels
import combo.sat.*
import combo.test.assertEquals
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

abstract class ObjectiveFunctionTest {
    abstract fun function(nbrVariables: Int): ObjectiveFunction

    @Test
    fun valueAndImprovement() {
        for (p in TestModels.UNSAT_PROBLEMS + TestModels.SAT_PROBLEMS + TestModels.LARGE_PROBLEMS) {
            val f = function(p.nbrValues)
            val instance = BitArray(p.nbrValues)
            WordRandomSet().initialize(instance, Tautology, Random, null)
            val s = Validator(p, instance, Tautology)
            val v = f.value(instance)
            val ix = Random.nextInt(p.nbrValues)
            val imp = f.improvement(instance, ix)
            s.flip(ix)
            val nv = f.value(instance)
            assertEquals(imp, v - nv, 1E-3f)
        }
    }

    @Test
    fun upperBoundExhaustive() {
        val func = function(5)
        for (i in 0 until 32) {
            val instance = BitArray(5, intArrayOf(i))
            assertTrue(func.value(instance) - func.upperBound() <= 1E-6, "${func.value(instance)} <= ${func.upperBound()}")
        }
    }

    @Test
    fun lowerBoundExhaustive() {
        val func = function(5)
        for (i in 0 until 32) {
            val instance = BitArray(5, intArrayOf(i))
            assertTrue(func.value(instance) - func.lowerBound() >= -1E-6, "${func.value(instance)} >= ${func.lowerBound()}")
        }
    }
}

/**
 * Simplest test function, most likely unimodal
 */
class OneMaxObjective(val nbrVariables: Int) : ObjectiveFunction {
    override fun value(vector: VectorView) = -vector.iterator().asSequence().count().toFloat()
    override fun lowerBound() = -nbrVariables.toFloat()
    override fun upperBound() = 0.0f
}

/**
 * JUMP is a slightly harder bimodal test function than OneMax
 */
class JumpObjective(val n: Int, val m: Int = (0.8 * n).toInt()) : ObjectiveFunction {
    override fun value(vector: VectorView): Float {
        val count = vector.iterator().asSequence().count()
        val jump = (if (count <= n - m || count == n) m + count
        else n - count).toFloat()
        return -jump
    }

    override fun lowerBound() = -(n + m).toFloat()
    override fun upperBound() = 0.0f
}

class InteractionObjective(weights: FloatArray) : ObjectiveFunction {
    val weights = FloatMatrix(Array(weights.size) { i ->
        FloatArray(weights.size) { j ->
            if (abs(i - j) <= 2) weights[i] * weights[j] else 0.0f
        }
    })

    override fun value(vector: VectorView) = (weights * vector).sum()

    private val lowerBound: Float
    private val upperBound: Float

    init {
        var lb = 0.0f
        var ub = 0.0f
        for (i in weights.indices) {
            for (j in weights.indices) {
                if (abs(i - j) <= 2) {
                    lb += min(0.0f, weights[i] * weights[j])
                    ub += max(0.0f, weights[i] * weights[j])
                }
            }
        }
        lowerBound = lb
        upperBound = ub
    }

    override fun lowerBound() = lowerBound
    override fun upperBound() = upperBound
}

class InteractionObjectiveTest : ObjectiveFunctionTest() {
    override fun function(nbrVariables: Int) = InteractionObjective(FloatArray(nbrVariables) {
        Random.nextNormal()
    })
}

class OneMaxObjectiveTest : ObjectiveFunctionTest() {
    override fun function(nbrVariables: Int) = OneMaxObjective(nbrVariables)
}

class SatObjectiveTest : ObjectiveFunctionTest() {
    override fun function(nbrVariables: Int) = SatObjective
}

class JumpObjectiveTest : ObjectiveFunctionTest() {
    override fun function(nbrVariables: Int) = JumpObjective(nbrVariables)
}

class LinearObjectiveTest : ObjectiveFunctionTest() {
    override fun function(nbrVariables: Int) =
            LinearObjective(Random.nextBoolean(), FloatVector(FloatArray(nbrVariables) { Random.nextNormal() }))

    @Test
    fun valueOfOnes() {
        val weights = FloatVector(FloatArray(8) { 1.0f })
        val max = LinearObjective(true, weights)
        val min = LinearObjective(false, weights)

        val ones = BitArray(8, intArrayOf(0xFF))
        val zeros = BitArray(8)

        assertEquals(-8.0f, max.value(ones), 0.0f)
        assertEquals(-0.0f, max.value(zeros), 0.0f)
        assertEquals(8.0f, min.value(ones), 0.0f)
        assertEquals(0.0f, min.value(zeros), 0.0f)
    }

    @Test
    fun valueOfRange() {
        val weights = FloatVector(FloatArray(4) { it.toFloat() })
        val min = LinearObjective(false, weights)
        val max = LinearObjective(true, weights)

        val ones = SparseBitArray(4).apply { setBits(0, 4, 0b1111) }
        val zeros = SparseBitArray(4)

        assertEquals(-6.0f, max.value(ones), 0.0f)
        assertEquals(-0.0f, max.value(zeros), 0.0f)
        assertEquals(6.0f, min.value(ones), 0.0f)
        assertEquals(0.0f, min.value(zeros), 0.0f)
    }
}

class DisjunctPenaltyTest {
    @Test
    fun outOfReach() {
        val penalty = DisjunctPenalty(LinearPenalty())
        val function = LinearObjective(false, FloatVector(floatArrayOf(-1.0f, 0.5f, 2.0f, 1.0f)))
        var min = Float.POSITIVE_INFINITY
        var max = Float.NEGATIVE_INFINITY
        for (l in 0 until 2.0.pow(4).toInt()) {
            val instance = BitArray(4, intArrayOf(l))
            min = min(min, function.value(instance))
            max = max(max, function.value(instance))
        }
        for (l in 0 until 2.0.pow(4).toInt()) {
            val instance = BitArray(4, intArrayOf(l))
            val v = function.value(instance)
            val p = penalty.penalty(v, 1, function.lowerBound(), function.upperBound())
            assertTrue(v + p > max)
        }
    }
}
