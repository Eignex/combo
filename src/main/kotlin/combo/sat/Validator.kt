package combo.sat

import combo.expressions.Constraint
import combo.util.*
import kotlin.random.Random

/**
 * This contains cached information about satisfied constraints used during
 * search by search-based methods,
 * [combo.sat.optimizers.GeneticAlgorithm] and
 * [combo.sat.optimizers.LocalSearch].
 */
class Validator private constructor(
    val problem: Problem,
    val instance: Instance,
    val assumption: Constraint,
    private val assumptionIxs: IntCollection,
    private val unsatisfied: IntHashSet,
    private val constraintCache: IntArray,
    rebuildIndex: Boolean
) : Instance by instance {

    constructor(
        problem: Problem,
        instance: Instance,
        assumption: Constraint
    ) : this(
        problem, instance, assumption,
        if (assumption.literals.isEmpty()) EmptyCollection
        else IntHashSet(nullValue = -1).apply {
            assumption.literals.forEach { add(it.toIx()) }
        },
        IntHashSet(nullValue = -1),
        IntArray(problem.nbrConstraints + 1),
        true
    )

    var totalUnsatisfied: Int = 0
        private set


    init {
        if (rebuildIndex)
            rebuildIndex()
    }

    fun randomUnsatisfied(rng: Random): Constraint {
        val constId = unsatisfied.random(rng)
        return if (constId == constraintCache.lastIndex) assumption
        else problem.constraints[constId]
    }

    fun improvement(ix: Int): Int {
        val assumptionImprovement =
            if (ix in assumptionIxs) improvementConst(
                ix,
                assumption,
                constraintCache.lastIndex
            )
            else 0
        return assumptionImprovement + problem.constraining(ix)
            .sumOf { constId ->
                val const = problem.constraints[constId]
                improvementConst(ix, const, constId)
            }
    }

    private fun improvementConst(
        ix: Int,
        const: Constraint,
        constId: Int
    ): Int {
        val oldFlips = const.violations(this, constraintCache[constId])
        instance.flip(ix)
        val cacheUpdate =
            const.cacheUpdate(constraintCache[constId], instance.literal(ix))
        val newFlips = const.violations(this, cacheUpdate)
        instance.flip(ix)
        return oldFlips - newFlips
    }

    override fun flip(ix: Int) = set(ix, !isSet(ix))

    override fun set(ix: Int, value: Boolean) {
        val literal = ix.toLiteral(value)
        if (instance.literal(literal.toIx()) == literal) return
        if (ix in assumptionIxs)
            updateConst(ix, assumption, constraintCache.lastIndex)
        for (constId in problem.constraining(ix)) {
            val const = problem.constraints[constId]
            updateConst(ix, const, constId)
        }
        instance.flip(ix)
    }

    private fun updateConst(ix: Int, const: Constraint, constId: Int) {
        val oldFlips = const.violations(this, constraintCache[constId])
        instance.flip(ix)
        constraintCache[constId] =
            const.cacheUpdate(constraintCache[constId], instance.literal(ix))
        val newFlips = const.violations(this, constraintCache[constId])
        instance.flip(ix)
        if (oldFlips > 0 && newFlips == 0) unsatisfied.remove(constId)
        else if (newFlips > 0 && oldFlips == 0) unsatisfied.add(constId)
        totalUnsatisfied += newFlips - oldFlips
    }

    override fun setWord(wordIx: Int, value: Int) {
        instance.setWord(wordIx, value)
        rebuildIndex()
    }

    override fun equals(other: Any?) = instance == other
    override fun hashCode() = instance.hashCode()

    override fun clear() {
        instance.clear()
        rebuildIndex()
    }

    override fun copy(): Validator {
        val copy = Validator(
            problem,
            instance.copy(),
            assumption,
            assumptionIxs,
            unsatisfied.copy(),
            constraintCache.copyOf(),
            false
        )
        copy.totalUnsatisfied = totalUnsatisfied
        return copy
    }

    private fun rebuildIndex() {
        totalUnsatisfied = 0
        unsatisfied.clear()
        for ((constId, const) in problem.constraints.withIndex()) {
            constraintCache[constId] = const.cache(instance)
            totalUnsatisfied += const.violations(this, constraintCache[constId])
                .also {
                    if (it > 0) unsatisfied.add(constId)
                }
        }
        constraintCache[constraintCache.lastIndex] = assumption.cache(instance)
        totalUnsatisfied += assumption.violations(this, constraintCache.last())
            .also {
                if (it > 0) unsatisfied.add(constraintCache.lastIndex)
            }
    }
}
