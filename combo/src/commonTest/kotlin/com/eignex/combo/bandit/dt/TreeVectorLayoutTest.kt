package com.eignex.combo.bandit.dt

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.test.TimedTest
import com.eignex.klause.schema.and
import com.eignex.klause.solver.Sample
import com.eignex.kumulant.bandit.univariate.Greedy
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class TreeVectorLayoutTest : TimedTest() {
    private class Model : DecisionSpace() {
        val flag by boolVar()
        val count by intVar(0, 10)
        val value by floatVar(0.0, 1.0)
        val mode by nominal("slow", "fast")
    }

    private fun sample(space: com.eignex.combo.decisions.CompiledDecisionSpace): Sample {
        val bools = BooleanArray(space.compiled.problem.numBoolVars)
        bools[space.compiled.nominalIndicators.getValue("mode").getValue("slow")] = true
        return Sample(bools, longArrayOf(3), doubleArrayOf(0.5))
    }

    @Test
    fun `vector routes match typed numeric nominal and expression predicates`() {
        val model = Model()
        val space = model.compileSpace()
        val candidates = listOf(
            BoolSplit(model.flag),
            IntThresholdSplit(model.count, 3),
            IntThresholdSplit(model.count, 7),
            FloatThresholdSplit(model.value, 0.333),
            FloatThresholdSplit(model.value, 0.777),
            NominalSplit(model.mode, "fast"),
            ExprSplit(model.flag and (model.mode eq "fast")),
        )
        val layout = TreeVectorLayout(candidates)
        val base = sample(space)
        val raw = Sample(base.bools, base.ints, doubleArrayOf(0.543210987))
        val row = TreeFeatureProjection(space).encode(BanditSample(raw))
        val features = layout.encode(row)
        assertEquals(5, features.size)
        assertEquals(0.543210987, features[2], 1e-12)
        candidates.zip(layout.splits).forEach { (typed, vector) ->
            assertEquals(typed.direction(row), vector.direction(features))
            assertEquals(typed, layout.predicate(vector))
        }
        assertEquals(features.toDoubleArray().toList(), (0 until features.size).map { features[it] })
        assertFailsWith<IndexOutOfBoundsException> { features[-1] }
        assertFailsWith<IndexOutOfBoundsException> { features[features.size] }
    }

    @Test
    fun `empty predicate pools retain a valid constant vector`() {
        val space = Model().compileSpace()
        val sample = BanditSample(sample(space))
        val layout = TreeVectorLayout(emptyList())
        assertEquals(listOf(0.0), layout.encode(TreeFeatureProjection(space).encode(sample)).toDoubleArray().toList())
    }

    @Test
    fun `forest snapshots transfer learned state and remain independent of later updates`() {
        val model = Model()
        val space = model.compileSpace()
        val raw = sample(space)
        val sample = BanditSample(raw)
        val original = RandomForestBandit.build(
            space,
            policy = Greedy(priorWeight = 0.0),
            nbrTrees = 3,
            bagging = false,
            randomSeed = 3,
        )
        repeat(2) { original.train(sample, 7.0) }
        val data = original.exportData()
        original.train(sample, 100.0)
        val replica = RandomForestBandit.build(
            space,
            policy = Greedy(priorWeight = 0.0),
            nbrTrees = 3,
            bagging = false,
            randomSeed = 4,
        )
        replica.importData(data)
        assertEquals(7.0, replica.predict(sample), 1e-12)
        assertTrue(original.predict(sample) > replica.predict(sample))
        replica.regression.reset()
        assertEquals(0.0, replica.trees[0].predict(TreeFeatureProjection(space).encode(sample)))
    }

    @Test
    fun `single tree snapshots transfer learned state`() {
        val space = Model().compileSpace()
        val sample = BanditSample(sample(space))
        val original = DecisionTreeBandit(space, Greedy(priorWeight = 0.0), { _, _ -> sample.sample })
        repeat(2) { original.train(sample, 7.0) }
        val replica = DecisionTreeBandit(space, Greedy(priorWeight = 0.0), { _, _ -> sample.sample })
        replica.importData(original.exportData())
        assertEquals(7.0, replica.predict(sample), 1e-12)
    }
}
