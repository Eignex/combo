package combo.bandit.dt

import com.eignex.klause.solver.LocalSearchParams
import com.eignex.klause.solver.LocalSearchSolver
import combo.decisions.DecisionSpace
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

private class ToyTree : DecisionSpace() {
    val premium by boolVar()
    val budget  by intVar(0, 1000)
    val tier    by nominal("free", "pro", "enterprise")
}

class TreeFeatureProjectionTest {

    @Test
    fun `encode should yield a row that decodes by handle`() {
        val model = ToyTree()
        val space = model.compileSpace()
        val projection = TreeFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)
        val sample = solver.sample(LocalSearchParams(randomSeed = 1L))!!

        val row = projection.encode(combo.decisions.BanditSample.undithered(sample))

        // Each handle dispatches to klause's typed decode under the hood.
        assertEquals(space.compiled.decode(model.premium, sample), row.bool(model.premium))
        assertEquals(space.compiled.decode(model.budget, sample), row.int(model.budget))
        assertEquals(space.compiled.decode(model.tier, sample), row.nominal(model.tier))
    }

    @Test
    fun `bool split should route by the handle's value`() {
        val model = ToyTree()
        val space = model.compileSpace()
        val projection = TreeFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)
        val split = BoolSplit(model.premium)

        // Try multiple samples — directions must agree with the underlying bool.
        repeat(10) { seed ->
            val s = solver.sample(LocalSearchParams(randomSeed = seed.toLong()))!!
            val row = projection.encode(combo.decisions.BanditSample.undithered(s))
            assertEquals(row.bool(model.premium), split.direction(row))
        }
    }

    @Test
    fun `int threshold split should route by value le threshold`() {
        val model = ToyTree()
        val space = model.compileSpace()
        val projection = TreeFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)
        val split = IntThresholdSplit(model.budget, threshold = 500)

        repeat(20) { seed ->
            val s = solver.sample(LocalSearchParams(randomSeed = seed.toLong()))!!
            val row = projection.encode(combo.decisions.BanditSample.undithered(s))
            val expected = row.int(model.budget) <= 500
            assertEquals(expected, split.direction(row))
        }
    }

    @Test
    fun `nominal split should partition labels`() {
        val model = ToyTree()
        val space = model.compileSpace()
        val projection = TreeFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)
        val split = NominalSplit(model.tier, leftLabels = setOf("free"))

        repeat(20) { seed ->
            val s = solver.sample(LocalSearchParams(randomSeed = seed.toLong()))!!
            val row = projection.encode(combo.decisions.BanditSample.undithered(s))
            assertEquals(row.nominal(model.tier) == "free", split.direction(row))
        }
    }

    @Test
    fun `int split should reject thresholds outside the handle domain`() {
        val model = ToyTree()
        model.compileSpace()  // realise the handle.min/max
        assertFailsWith<IllegalArgumentException> {
            IntThresholdSplit(model.budget, threshold = 1500)
        }
    }

    @Test
    fun `nominal split should reject empty or full label partitions`() {
        val model = ToyTree()
        model.compileSpace()
        assertFailsWith<IllegalArgumentException> {
            NominalSplit(model.tier, leftLabels = emptySet())
        }
        assertFailsWith<IllegalArgumentException> {
            NominalSplit(model.tier, leftLabels = setOf("free", "pro", "enterprise"))
        }
    }

    @Test
    fun `featureSize counts klause bool plus int vars`() {
        val space = ToyTree().compileSpace()
        val projection = TreeFeatureProjection(space)
        // ToyTree: premium (1 bool) + tier (3 indicator bools) + budget (1 int) = 4 + 1 = 5
        assertEquals(5, projection.featureSize)
        assertEquals(4, space.compiled.problem.numBoolVars)
        assertEquals(1, space.compiled.problem.numIntVars)
    }

    @Test
    fun `isPresent should report active state on the row`() {
        val model = ToyTree()
        val space = model.compileSpace()
        val projection = TreeFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)
        val s = solver.sample(LocalSearchParams(randomSeed = 99L))!!
        val row = projection.encode(combo.decisions.BanditSample.undithered(s))
        // No optionals declared in ToyTree → everything is always present.
        assertTrue(row.isPresent(model.premium))
        assertTrue(row.isPresent(model.budget))
        assertTrue(row.isPresent(model.tier))
    }

}
