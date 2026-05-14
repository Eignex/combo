package combo.bandit.dt

import com.eignex.klause.ast.and
import com.eignex.klause.ast.gt
import com.eignex.klause.ast.not
import com.eignex.klause.ast.or
import com.eignex.klause.solver.localsearch.LocalSearchParams
import com.eignex.klause.solver.localsearch.LocalSearchSolver
import combo.decisions.BanditSample
import combo.decisions.DecisionSpace
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The split layer wraps every routing predicate in a klause [com.eignex.klause.ast.BoolExpr].
 * These tests verify that the typed cases produce expected expressions, [ExprSplit] accepts
 * arbitrary boolean combinations, and the pretty printer reads like the source DSL.
 */
private class SplitTestSpace : DecisionSpace() {
    val premium by boolVar()
    val budget by intVar(0, 1000)
    val tier by nominal("free", "pro", "enterprise")
}

class SplitTest {

    @Test
    fun `BoolSplit prints as the handle name`() {
        val model = SplitTestSpace()
        model.compileSpace()
        assertEquals("premium", BoolSplit(model.premium).toString())
    }

    @Test
    fun `IntThresholdSplit prints in infix form`() {
        val model = SplitTestSpace()
        model.compileSpace()
        assertEquals("budget <= 500", IntThresholdSplit(model.budget, 500).toString())
    }

    @Test
    fun `NominalSplit prints in equality form`() {
        val model = SplitTestSpace()
        model.compileSpace()
        assertEquals("tier == \"pro\"", NominalSplit(model.tier, "pro").toString())
    }

    @Test
    fun `ExprSplit routes by an arbitrary klause expression`() {
        val model = SplitTestSpace()
        val space = model.compileSpace()
        val solver = LocalSearchSolver(space.compiled.problem)

        // Composite predicate: premium AND budget > 500. Mixes a bool and an int compare
        // in one split — impossible with the typed cases alone, trivial here.
        val split = ExprSplit(model.premium.toExpr() and (model.budget gt 500))
        val projection = TreeFeatureProjection(space)

        repeat(20) { seed ->
            val s = solver.sample(LocalSearchParams(randomSeed = seed.toLong()))!!
            val row = projection.encode(BanditSample.undithered(s))
            val expected = row.bool(model.premium) && row.int(model.budget) > 500
            assertEquals(expected, split.direction(row))
        }
    }

    @Test
    fun `pretty prints nested boolean expressions readably`() {
        val model = SplitTestSpace()
        model.compileSpace()

        // (premium and !(tier == "free")) or (budget > 500)
        val expr = (model.premium.toExpr() and !(model.tier eq "free")) or (model.budget gt 500)
        val out = ExprSplit(expr).toString()

        // Verify the pretty printer surfaces the operator vocabulary the user wrote.
        assertTrue("premium" in out, "expected 'premium' in: $out")
        assertTrue("tier == \"free\"" in out, "expected 'tier == \"free\"' in: $out")
        assertTrue("budget > 500" in out, "expected 'budget > 500' in: $out")
        assertTrue(" or " in out, "expected ' or ' in: $out")
        assertTrue(" and " in out, "expected ' and ' in: $out")
    }
}
