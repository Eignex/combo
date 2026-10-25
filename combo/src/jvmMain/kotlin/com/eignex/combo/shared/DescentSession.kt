package com.eignex.combo.shared

import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.solver.Sample
import com.eignex.klause.solver.Session
import com.eignex.klause.solver.SolverParams
import kotlin.random.Random

/**
 * Incremental constraint-solving handle wrapping Klause's [Session] assumption stack.
 * [push] pins a scope, [sample] requests a feasible witness, and [pop] undoes the scope.
 * A null witness can mean a budget miss or proven infeasibility; callers retain an
 * incumbent until a replacement is verified feasible.
 */
interface DescentSession {
    /** Current depth of the assumption stack. */
    val depth: Int

    /** Pin a scope of assumptions on top of the current stack. */
    fun push(assumptions: Assumptions)

    /** Undo the most recent [push]. */
    fun pop()

    /** Solve under the current stack; null can mean a budget miss or proven infeasibility. */
    fun sample(rng: Random): Sample?
}

/**
 * Adapts a klause [Session] into a [DescentSession]. [paramsFor] mints fresh solve
 * parameters per attempt (typically seeding the backend RNG from the bandit's [Random]),
 * and [sample] unwraps the [com.eignex.klause.solver.result.SampleResult] into a nullable
 * witness via its `assignment`.
 */
class KlauseDescentSession<P : SolverParams>(private val session: Session<P>, private val paramsFor: (Random) -> P) :
    DescentSession {
    override val depth: Int get() = session.depth
    override fun push(assumptions: Assumptions) = session.push(assumptions)
    override fun pop() = session.pop()
    override fun sample(rng: Random): Sample? = session.sample(paramsFor(rng)).assignment
}
