package combo.sat.optimizers

import combo.expressions.Cardinality
import combo.expressions.Conjunction
import combo.expressions.Disjunction
import combo.expressions.Linear
import combo.expressions.ReifiedEquivalent
import combo.expressions.ReifiedImplies
import combo.math.toIntArray
import combo.expressions.Constraint
import combo.expressions.Empty
import combo.expressions.Tautology
import combo.sat.*
import combo.expressions.Relation.*
import combo.util.IntCollection
import combo.util.RandomSequence
import org.sat4j.core.LiteralsUtils.negLit
import org.sat4j.core.LiteralsUtils.posLit
import org.sat4j.core.Vec
import org.sat4j.core.VecInt
import org.sat4j.minisat.SolverFactory
import org.sat4j.minisat.core.IPhaseSelectionStrategy
import org.sat4j.pb.OptToPBSATAdapter
import org.sat4j.pb.PseudoOptDecorator
import org.sat4j.pb.core.PBSolver
import org.sat4j.specs.ContradictionException
import org.sat4j.specs.ISolver
import org.sat4j.specs.ISolverService
import org.sat4j.tools.ModelIterator
import org.sat4j.tools.SearchListenerAdapter
import java.math.BigInteger
import java.math.BigInteger.valueOf
import kotlin.random.Random
import org.sat4j.minisat.core.Solver as Sat4J
import org.sat4j.pb.ObjectiveFunction as Sat4JObjectiveFunction
import org.sat4j.pb.SolverFactory as PBSolverFactory

/**
 * [Optimizer] of [LinearObjective] using the Sat4J SAT library. Using this requires an extra optional
 * dependency. The internal solver is re-used for [witness] and [asSequence] and kept as thread local.
 * For [optimize] a new one is created for each call due to limitations in Sat4J.
 * @param problem the problem contains the [Constraint]s and the number of variables.
 * @param randomSeed Set the random seed to a specific value to have a reproducible algorithm.
 * @param timeout The solver will abort after timeout in milliseconds have been reached, without a real-time guarantee.
 * @param instanceFactory Determines the [Instance] that will be created for solving.
 * @param maxConflicts Solver aborts after this number of conflicts are reached.
 * @param delta Precision with which to convert objective function into integer constraints.
 * @param gcdSimplify Simplify weights with greatest common divisor before optimizing.
 */
class Sat4JSolver @JvmOverloads constructor(
        val problem: Problem,
        override val randomSeed: Int = System.currentTimeMillis().toInt(),
        override val timeout: Long = -1L,
        val instanceFactory: InstanceFactory = BitArrayFactory,
        val maxConflicts: Int = 0,
        val delta: Float = 0.01f,
        val gcdSimplify: Boolean = true,
        val constraintHandler: (Constraint, ISolver) -> Unit = { c, _ ->
            throw UnsupportedOperationException("Constraint $c cannot be SAT encoded. " +
                    "Register custom constraint handler in order to handle extra constraints.")
        }) : Optimizer<LinearObjective> {

    private val randomSequence = RandomSequence(randomSeed)

    override val complete get() = true

    private val solverTL = ThreadLocal.withInitial {
        val solver: Sat4J<*> = if (problem.constraints.any { it is Linear }) PBSolverFactory.newLight() as Sat4J<*>
        else SolverFactory.newLight() as Sat4J<*>
        setupSolver(solver)
        solver
    }

    private fun setupSolver(solver: ISolver) {

        solver.setKeepSolverHot(true)
        solver.newVar(problem.nbrValues)

        try {
            for (c in problem.constraints)
                when (c) {
                    is Disjunction -> solver.addClause(VecInt(c.literals.toArray()))
                    is Conjunction -> c.literals.forEach {
                        solver.addClause(VecInt(intArrayOf(it)))
                    }
                    is Cardinality -> {
                        val cardLits = VecInt(c.literals.toArray())
                        when (c.relation) {
                            GE -> solver.addAtLeast(cardLits, c.degree)
                            LE -> solver.addAtMost(cardLits, c.degree)
                            EQ -> solver.addExactly(cardLits, c.degree)
                            GT -> solver.addAtLeast(cardLits, c.degree + 1)
                            LT -> solver.addAtMost(cardLits, c.degree - 1)
                            NE ->
                                throw UnsupportedOperationException("Relation != cannot be expressed as a linear inequality.")
                        }
                    }
                    is Linear -> {
                        val pbSolver = solver as PBSolver
                        val cardLits = VecInt(c.literals.toArray())
                        val weights = Vec(Array<BigInteger>(cardLits.size()) {
                            valueOf(c.weights[c.literals[cardLits[it]]].toLong())
                        })
                        when (c.relation) {
                            GE -> pbSolver.addPseudoBoolean(cardLits, weights, true, valueOf(c.degree.toLong()))
                            LE -> pbSolver.addPseudoBoolean(cardLits, weights, false, valueOf(c.degree.toLong()))
                            EQ -> {
                                pbSolver.addPseudoBoolean(cardLits, weights, true, valueOf(c.degree.toLong()))
                                pbSolver.addPseudoBoolean(cardLits, weights, false, valueOf(c.degree.toLong()))
                            }
                            GT -> pbSolver.addPseudoBoolean(cardLits, weights, true, valueOf(c.degree.toLong() + 1L))
                            LT -> pbSolver.addPseudoBoolean(cardLits, weights, false, valueOf(c.degree.toLong() - 1L))
                            NE ->
                                throw UnsupportedOperationException("Relation != cannot be expressed as a linear inequality.")
                        }

                    }
                    is ReifiedEquivalent -> c.toCnf().forEach { solver.addClause(VecInt(it.literals.toArray())) }
                    is ReifiedImplies -> c.toCnf().forEach { solver.addClause(VecInt(it.literals.toArray())) }
                    is Tautology -> {
                    }
                    is Empty -> throw UnsatisfiableException("Empty constraint in problem.")
                    else -> constraintHandler.invoke(c, solver)
                }
        } catch (e: ContradictionException) {
            throw UnsatisfiableException("Failed to construct Sat4J problem.", cause = e)
        }

        if (maxConflicts > 0) solver.setTimeoutOnConflicts(maxConflicts)
        else solver.setTimeoutOnConflicts(Int.MAX_VALUE)
        // This disables Sat4j millisecond timeouts which spawns a thread for each solving

        for (i in 1..problem.nbrValues)
            solver.registerLiteral(i)

    }

    override fun witnessOrThrow(assumptions: IntCollection, guess: Instance?): Instance {
        val solver = solverTL.get()

        if (maxConflicts > 0) solver.setTimeoutOnConflicts(maxConflicts)
        else solver.setTimeoutOnConflicts(Int.MAX_VALUE)

        if (timeout > 0L) solver.setSearchListener(TimeoutListener(timeout))
        else solver.setSearchListener(VoidListener)

        if (guess == null)
            solver.order.phaseSelectionStrategy = RandomLiteralSelectionStrategySeeded(randomSequence.next())
        else solver.order.phaseSelectionStrategy = InitialGuessSelectionStrategy(guess)

        val assumption = VecInt(assumptions.toArray())
        try {
            if (solver.isSatisfiable(assumption)) return solver.model().toInstance()
            else throw UnsatisfiableException()
        } catch (e: org.sat4j.specs.TimeoutException) {
            throw IterationsReachedException(maxConflicts)
        }
    }

    override fun asSequence(assumptions: IntCollection): Sequence<Instance> {
        val solver = solverTL.get()

        if (maxConflicts > 0) solver.setTimeoutOnConflicts(maxConflicts)
        else solver.setTimeoutOnConflicts(Int.MAX_VALUE)

        if (timeout > 0L) solver.setSearchListener(TimeoutListener(timeout))
        else solver.setSearchListener(VoidListener)

        solver.order.phaseSelectionStrategy = RandomLiteralSelectionStrategySeeded(randomSequence.next())

        val iterator = ModelIterator(solver)
        val assumption = VecInt(assumptions.toArray())
        val timeout = timeout
        val end = if (timeout > 0L) System.currentTimeMillis() + timeout else Long.MAX_VALUE
        return generateSequence {
            try {
                if (timeout > 0L && System.currentTimeMillis() >= end) null
                else if (!iterator.isSatisfiable(assumption)) null
                else iterator.model().toInstance()
            } catch (e: TimeoutException) {
                null
            } catch (e: org.sat4j.specs.TimeoutException) {
                null
            }
        }
    }

    override fun optimizeOrThrow(function: LinearObjective, assumptions: IntCollection, guess: Instance?): Instance {
        val pbSolver = PBSolverFactory.newLight() as PBSolver
        setupSolver(pbSolver)

        if (timeout > 0L) pbSolver.setSearchListener(TimeoutListener(timeout))

        val intWeights = function.weights.toIntArray(delta, gcdSimplify)
        val bigIntWeights = Vec(Array<BigInteger>(function.weights.size) {
            val i = if (function.maximize) -intWeights[it] else intWeights[it]
            valueOf(i.toLong())
        })
        val variableLiterals = VecInt(IntArray(bigIntWeights.size()) { it + 1 })
        pbSolver.objectiveFunction = Sat4JObjectiveFunction(variableLiterals, bigIntWeights)
        if (guess != null) pbSolver.order.phaseSelectionStrategy = InitialGuessSelectionStrategy(guess)

        val assumption = VecInt(assumptions.toArray())
        val pseudoOptDecorator = PseudoOptDecorator(pbSolver)
        val optimizer = OptToPBSATAdapter(pseudoOptDecorator)
        try {
            if (optimizer.isSatisfiable(assumption))
                return optimizer.model().toInstance()
            else throw UnsatisfiableException()
        } catch (e: org.sat4j.specs.TimeoutException) {
            throw IterationsReachedException(maxConflicts)
        } finally {
            pbSolver.reset()
        }
    }

    private class TimeoutListener(val timeout: Long) : SearchListenerAdapter<ISolverService>() {
        private val end = timeout + System.currentTimeMillis()
        override fun assuming(p: Int) {
            if (end < System.currentTimeMillis()) throw TimeoutException(timeout)
        }
    }

    private object VoidListener : SearchListenerAdapter<ISolverService>() {
        private fun readResolve(): Any = VoidListener
    }

    private class RandomLiteralSelectionStrategySeeded(val rng: Random) : IPhaseSelectionStrategy {
        override fun assignLiteral(p: Int) {}
        override fun init(nlength: Int) {}
        override fun init(v: Int, p: Int) {}
        override fun updateVar(p: Int) {}
        override fun updateVarAtDecisionLevel(q: Int) {}
        override fun select(v: Int) = if (rng.nextBoolean()) posLit(v) else negLit(v)
    }

    private class InitialGuessSelectionStrategy(val guess: Instance) : IPhaseSelectionStrategy {
        override fun assignLiteral(p: Int) {}
        override fun init(nlength: Int) {}
        override fun init(v: Int, p: Int) {}
        override fun updateVarAtDecisionLevel(q: Int) {}
        override fun updateVar(p: Int) {}
        override fun select(v: Int) = if (guess.isSet(v - 1)) posLit(v) else negLit(v)
    }

    private fun IntArray.toInstance(): Instance {
        val create = instanceFactory.create(size)
        create.setAll(this)
        return create
    }
}
