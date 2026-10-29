package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.decisions.Scale
import com.eignex.combo.decisions.context
import com.eignex.combo.shared.WitnessBatch
import com.eignex.combo.shared.WitnessSearch
import com.eignex.combo.shared.WitnessTermination
import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.backtrack.BacktrackSolver
import com.eignex.klause.propagation.bake
import com.eignex.klause.schema.implies
import com.eignex.klause.schema.le
import com.eignex.klause.schema.not
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.Sample
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class InitialDesignTest {
    private class Rate : DecisionSpace() {
        val x by floatVar(0.0, 1.0)
    }

    private val rateWitnesses = WitnessSearch { _, _ ->
        WitnessBatch(
            listOf(
                Sample(
                    booleanArrayOf(),
                    longArrayOf(),
                    doubleArrayOf(0.5),
                    exactReals = listOf(BigFraction.ofDouble(0.5)!!),
                ),
            ),
            WitnessTermination.CountReached,
        )
    }

    @Test
    fun `Latin hypercube covers one stratum per point without fitting or acquisition scoring`() {
        val model = Rate()
        val space = model.compileSpace()
        val design = LatinHypercubeInitialDesign(space, 8, seed = 11, witnesses = rateWitnesses)
        val optimizer = BayesianOptimizer(
            SearchTransform(space),
            ForestSurrogate(),
            initialDesign = design,
            acquisitionOptimizer = AcquisitionOptimizer { _, _, _ ->
                error(
                    "initial design must bypass acquisition search",
                )
            },
        )
        assertEquals(2, design.size)
        val batch = optimizer.suggest(2)
        assertEquals(2, batch.size)
        assertEquals((0 until 2).toSet(), batch.map { (it.sample.float(model.x, space) * 2).toInt() }.toSet())
        assertEquals(0L, optimizer.fitCount)
        assertEquals(0, optimizer.lastSuggestion!!.acquisitionEvaluations)
    }

    @Test
    fun `structured design position survives transfer and counts pending configuration slots`() {
        val model = Rate()
        val space = model.compileSpace()
        fun optimizer() = BayesianOptimizer(
            SearchTransform(space),
            ForestSurrogate(),
            initialDesign = LatinHypercubeInitialDesign(space, 32, seed = 23, witnesses = rateWitnesses),
        )
        val left = optimizer()
        val first = left.suggest(3)
        left.complete(first[0].id, 1.0)
        left.train(first[0].sample, 2.0)
        val right = optimizer()
        right.importData(left.exportData())
        assertEquals(left.suggest(1)[0].sample, right.suggest(1)[0].sample)
    }

    @Test
    fun `random exploration uses a dedicated witness source without fitting or local search`() {
        val space = Rate().compileSpace()
        val raw = Sample(
            booleanArrayOf(),
            longArrayOf(),
            doubleArrayOf(0.25),
            exactReals = listOf(BigFraction.ofDouble(0.25)!!),
        )
        val optimizer = BayesianOptimizer(
            SearchTransform(space),
            ForestSurrogate(),
            initialDesign = RandomInitialDesign(probability = 1.0),
            acquisitionOptimizer = AcquisitionOptimizer { _, _, _ -> error("random exploration used local search") },
            exploration = WitnessSearch { _, _ -> WitnessBatch(listOf(raw), WitnessTermination.CountReached) },
        )
        assertEquals(BanditSample(raw), optimizer.suggest(1)[0].sample)
        assertEquals(0L, optimizer.fitCount)
    }
    private class Mixed : DecisionSpace() {
        val rate by floatVar(1e-6, 1.0, scale = Scale.Log)
        val count by intVar(1, 6)
        val premium by contextBool()
        val cap by constraint { !premium implies (count le 3) }
    }

    @Test
    fun `constrained mixed designs retain hard context and source units`() {
        val model = Mixed()
        val space = model.compileSpace()
        val ctx = context { set(model.premium, false) }
        val anchor = BacktrackSolver(space.compiled.problem.bake()).sample(
            BacktrackParams(maxDecisions = 32, assumptions = space.assumptionsFor(ctx), randomSeed = 5),
        ).assignment!!
        val optimizer = BayesianOptimizer(
            SearchTransform(space),
            ForestSurrogate(),
            initialDesign = LatinHypercubeInitialDesign(
                space,
                4,
                seed = 5,
                witnesses = WitnessSearch { _, _ ->
                    WitnessBatch(listOf(anchor), WitnessTermination.CountReached)
                },
            ),
        )
        val batch = optimizer.suggest(1, ctx)
        assertEquals(1, batch.size)
        assertTrue(batch.all { space.matches(it.sample.sample, space.assumptionsFor(ctx)) })
        assertTrue(batch.all { space.compiled.decode(model.count, it.sample.sample) in 1L..3L })
        assertTrue(batch.all { it.sample.float(model.rate, space) in 1e-6..1.0 })
    }

    @Test
    fun `initial configuration budget accounts for planned repeated measurements`() {
        val space = Rate().compileSpace()
        val design = LatinHypercubeInitialDesign(
            space,
            32,
            measurementsPerConfiguration = 3.0,
            witnesses = rateWitnesses,
        )
        assertEquals(2, design.size)
        val optimizer = BayesianOptimizer(SearchTransform(space), ForestSurrogate(), initialDesign = design)
        assertEquals(2, optimizer.suggest(2).size)
        assertEquals(0L, optimizer.fitCount)
    }
}
