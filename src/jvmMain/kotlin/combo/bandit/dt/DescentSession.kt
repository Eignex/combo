package combo.bandit.dt

import com.eignex.klause.solver.Assumptions
import com.eignex.klause.solver.Sample
import com.eignex.klause.solver.Session
import com.eignex.klause.solver.SolverParams
import kotlin.random.Random

/**
 * Incremental constraint-solving handle the forest drives during its greedy descent.
 * Wraps klause's [Session] assumption stack: [push] pins a decision, [sample] asks the
 * backend for a feasible witness under the current stack (null when over-constrained),
 * and [pop] undoes the most recent push.
 *
 * The session-based descent only ever *commits* pins it has verified feasible (every
 * kept pin came back with a witness), so the accumulated stack is always satisfiable —
 * no conflict-directed backjumping is needed, just a local [pop] when a candidate pin
 * turns the problem infeasible. For problems the iterative solver can't crack (the base
 * problem itself returns no witness), the forest falls back to the propagation-based
 * CDCL descent.
 */
interface DescentSession {
    /** Current depth of the assumption stack. */
    val depth: Int

    /** Pin a scope of assumptions on top of the current stack. */
    fun push(assumptions: Assumptions)

    /** Undo the most recent [push]. */
    fun pop()

    /** Solve under the current stack; a feasible witness or null when over-constrained. */
    fun sample(rng: Random): Sample?
}

/**
 * Adapts a klause [Session] into a [DescentSession]. [paramsFor] mints fresh solve
 * parameters per attempt (typically seeding the backend RNG from the bandit's [Random]),
 * and [sample] unwraps the [com.eignex.klause.solver.SampleResult] into a nullable
 * witness via its `assignment`.
 */
class KlauseDescentSession<P : SolverParams>(
    private val session: Session<P>,
    private val paramsFor: (Random) -> P,
) : DescentSession {
    override val depth: Int get() = session.depth
    override fun push(assumptions: Assumptions) = session.push(assumptions)
    override fun pop() = session.pop()
    override fun sample(rng: Random): Sample? = session.sample(paramsFor(rng)).assignment
}
