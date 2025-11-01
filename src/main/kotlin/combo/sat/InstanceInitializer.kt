package combo.sat

import combo.math.nextGeometric
import combo.math.nextNormal
import combo.math.permutation
import combo.expressions.Constraint
import combo.expressions.Cardinality
import combo.expressions.Disjunction
import combo.expressions.Relation
import combo.sat.optimizers.LinearObjective
import combo.sat.optimizers.ObjectiveFunction
import kotlin.jvm.JvmOverloads
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt
import kotlin.random.Random

enum class InitializerType {

    /**
     * Set all to false.
     */
    NONE,

    /**
     * Used for linear objective only. It makes a guess based on the weight of each variable.
     */
    WEIGHT_MAX,

    /**
     * Using uniformly random guess.
     */
    RANDOM,

    /**
     * Starts with random guess and the coerces each constraint in order to be satisfied.
     */
    COERCE,

    /**
     * Starts with random guess, then follows implications, and then coerces constraints to be satisfied.
     */
    PROPAGATE_COERCE,

    /**
     * Starts with WEIGHT_MAX then applies PROPAGATE_COERCE.
     */
    WEIGHT_MAX_PROPAGATE_COERCE
}

interface InstanceInitializer<in O : ObjectiveFunction?> {
    fun initialize(mi: Instance, assumption: Constraint, rng: Random, function: O)
}

class ConstraintCoercer<in O : ObjectiveFunction?>(val problem: Problem, val randomizer: InstanceInitializer<O>) : InstanceInitializer<O> {

    private val prioritizedConstraints = problem.constraints.copyOf().apply {
        sortWith(Comparator { a, b ->
            a.priority.compareTo(b.priority)
        })
    }

    override fun initialize(mi: Instance, assumption: Constraint, rng: Random, function: O) {
        randomizer.initialize(mi, assumption, rng, function)
        assumption.coerce(mi, rng)
        for (c in prioritizedConstraints) c.coerce(mi, rng)
    }
}

class ImplicationConstraintCoercer<in O : ObjectiveFunction?>(val problem: Problem,
                                                              val transitiveImplications: TransitiveImplications,
                                                              val randomizer: InstanceInitializer<O>) : InstanceInitializer<O> {

    private fun is2sat(constraint: Constraint) = (constraint is Disjunction && constraint.literals.size == 2) ||
            (constraint is Cardinality && constraint.degree == 1 && constraint.relation == Relation.LE) ||
            (constraint is Cardinality && constraint.degree == 2 && constraint.relation == Relation.LT)

    val complexConstraints = problem.constraints.filter { !is2sat(it) }.toTypedArray().apply {
        sortWith(Comparator { a, b ->
            a.priority.compareTo(b.priority)
        })
    }

    override fun initialize(mi: Instance, assumption: Constraint, rng: Random, function: O) {
        randomizer.initialize(mi, assumption, rng, function)
        assumption.coerce(mi, rng)
        for (i in permutation(problem.nbrValues, rng)) {
            val lit = mi.literal(i)
            transitiveImplications.propagate(lit, mi)
        }
        for (c in complexConstraints)
            c.coerce(mi, rng)
    }
}

class NoInitializer(val assumptions: Boolean) : InstanceInitializer<ObjectiveFunction?> {
    override fun initialize(mi: Instance, assumption: Constraint, rng: Random, function: ObjectiveFunction?) {
        if (assumptions) assumption.coerce(mi, rng)
    }
}

class WeightSet @JvmOverloads constructor(val noiseStd: Float = 0.5f) : InstanceInitializer<LinearObjective> {
    override fun initialize(mi: Instance, assumption: Constraint, rng: Random, function: LinearObjective) {
        for (i in mi.indices) {
            mi[i] = if (function.maximize) rng.nextNormal(function.weights[i], noiseStd) >= 0.0f
            else rng.nextNormal(function.weights[i], noiseStd) < 0.0f
        }
    }
}

class RandomSet(val pBias: Float = .5f) : InstanceInitializer<ObjectiveFunction?> {
    override fun initialize(mi: Instance, assumption: Constraint, rng: Random, function: ObjectiveFunction?) {
        for (j in mi.indices) mi[j] = rng.nextFloat() < pBias
    }
}

/**
 * Initializes an [Instance] a whole word (32 bit int) at a time. This approximates pBias to the closest multiple of
 * 1/2^n. For n=1, i.e. pBias = 0.5f the speedup compared to[RandomSet] is about 50x.
 * For n=3 (pBias=0.125 or pBias=0.875) the speedup is 25x.
 */
class WordRandomSet(val start: Int, val steps: Int) : InstanceInitializer<ObjectiveFunction?> {

    constructor(pBias: Float = .5f) : this(if (pBias > 0.5f) 0 else -1, min(sqrt(1 / min(pBias, 1 - pBias)).roundToInt(), 8))

    override fun initialize(mi: Instance, assumption: Constraint, rng: Random, function: ObjectiveFunction?) {
        val misaligned = mi.size and 0x1F > 0
        val ints = (mi.size shr 5) + if (misaligned) 1 else 0
        var offset = 0
        for (i in 0 until ints) {
            var k = start
            for (j in 1..steps) {
                if (start == 0) {
                    k = k or rng.nextInt()
                    if (k == -1) break
                } else {
                    k = k and rng.nextInt()
                    if (k == 0) break
                }
            }
            if (k == 0) continue
            if (i == ints - 1 && misaligned) {
                val nbrBits = mi.size and 0x1F
                mi.setBits(offset, nbrBits, k and (-1 shl nbrBits).inv())
            } else mi.setBits(offset, 32, k)
            offset += 32
        }
    }
}

/**
 * Useful for very sparse problems where [pBias] is close to 0.
 */
class GeometricRandomSet(val pBias: Float = 0.5f) : InstanceInitializer<ObjectiveFunction?> {
    override fun initialize(mi: Instance, assumption: Constraint, rng: Random, function: ObjectiveFunction?) {
        var index = rng.nextGeometric(pBias) - 1
        while (index < mi.size) {
            mi.flip(index)
            index += rng.nextGeometric(pBias)
        }
    }
}
