package com.eignex.combo.bo

import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.shared.WitnessBatch
import com.eignex.combo.shared.WitnessSearch
import com.eignex.combo.shared.WitnessTermination
import com.eignex.klause.schema.not
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.Sample
import com.eignex.koblas.DenseVector
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class AcquisitionSearchTest {
    private class Rate : DecisionSpace() {
        val x by floatVar(0.0, 1.0)
    }
    private fun point(x: Double) = Sample(
        booleanArrayOf(),
        longArrayOf(),
        doubleArrayOf(x),
        exactReals = listOf(BigFraction.ofDouble(x)!!),
    )
    private val zero = Acquisition { rows -> DenseVector.zero(rows.rows) }

    @Test
    fun `seeded acquisition ties do not always select the first history anchor`() {
        val projection = SearchTransform(Rate().compileSpace())
        val backend = WitnessSearch { _, _ ->
            WitnessBatch(List(20) { point(0.5) } + listOf(point(0.0), point(1.0)), WitnessTermination.CountReached)
        }
        val search = LocalAndRandomSearch(projection, backend, localSteps = 0)
        val request = AcquisitionRequest(Context.Empty, emptyList(), listOf(point(0.5)))
        val selections = (0 until 30).map { search.maximize(zero, request, Random(it)).sample!!.sample.reals[0] }
        assertEquals(setOf(0.0, 0.5, 1.0), selections.toSet())
        assertEquals(
            selections,
            (0 until 30).map { search.maximize(zero, request, Random(it)).sample!!.sample.reals[0] },
        )
        val first = LocalAndRandomSearch(projection, backend, localSteps = 0, tieBreak = AcquisitionTieBreak.First)
        assertEquals(0.5, first.maximize(zero, request, Random(1)).sample!!.sample.reals[0])
    }

    @Test
    fun `tie breaking preserves strictly better acquisition values`() {
        val projection = SearchTransform(Rate().compileSpace())
        val backend = WitnessSearch { _, _ ->
            WitnessBatch(listOf(point(0.0), point(0.5), point(1.0)), WitnessTermination.CountReached)
        }
        val search = LocalAndRandomSearch(projection, backend, localSteps = 0)
        val request = AcquisitionRequest(Context.Empty, emptyList(), listOf(point(0.0)))
        repeat(10) {
            val selected = search.maximize(
                Acquisition { rows -> DenseVector.of(DoubleArray(rows.rows) { rows[it, 0] }) },
                request,
                Random(it),
            )
            assertEquals(1.0, selected.sample!!.sample.reals[0])
            assertEquals(3, selected.evaluations)
        }
    }

    @Test
    fun `excluded anchors still seed novel feasible interiors and are scored only once`() {
        val projection = SearchTransform(Rate().compileSpace())
        val seeds = listOf(point(0.0), point(1.0))
        val backend = WitnessSearch { _, _ -> WitnessBatch(seeds, WitnessTermination.CountReached) }
        val searched = mutableListOf<Double>()
        val selected = LocalAndRandomSearch(projection, backend, localSteps = 2).maximize(
            Acquisition { rows ->
                searched += List(rows.rows) { rows[it, 0] }
                DenseVector.of(DoubleArray(rows.rows) { -(rows[it, 0] - 0.5) * (rows[it, 0] - 0.5) })
            },
            AcquisitionRequest(Context.Empty, seeds, seeds),
            Random(7),
        )
        assertEquals(0.5, selected.sample!!.sample.reals[0])
        assertEquals(searched.size, searched.distinct().size)
        assertTrue(0.0 in searched && 1.0 in searched)
    }

    @Test
    fun `multiple starts escape a better initial local peak`() {
        val projection = SearchTransform(Rate().compileSpace())
        val backend = WitnessSearch { _, _ ->
            WitnessBatch(listOf(point(0.2), point(0.8)), WitnessTermination.CountReached)
        }
        val acquisition = Acquisition { rows ->
            DenseVector.of(
                DoubleArray(rows.rows) {
                    val x = rows[it, 0]
                    maxOf(
                        1.0 - 100.0 * (x - 0.2) * (x - 0.2),
                        2.0 - 1000.0 * (x - 0.9) * (x - 0.9),
                    )
                },
            )
        }
        fun select(starts: Int) = LocalAndRandomSearch(
            projection,
            backend,
            localSteps = 8,
            neighbors = 64,
            localPolicy = LocalSearchPolicy(starts = starts, initialRadius = 0.12),
        ).maximize(
            acquisition,
            AcquisitionRequest(Context.Empty, emptyList(), emptyList()),
            Random(7),
        )
        val single = select(1).sample!!.sample.reals[0]
        val multiple = select(2).sample!!.sample.reals[0]
        assertTrue(kotlin.math.abs(single - 0.2) < 0.01)
        assertTrue(kotlin.math.abs(multiple - 0.9) < 0.01)
    }

    private class IntegerRate : DecisionSpace() {
        val x by intVar(
            1,
            128,
            scale = com.eignex.combo.decisions.Scale.Log,
        )
    }

    @Test
    fun `integer neighborhoods move in declared logarithmic geometry`() {
        val model = IntegerRate()
        val space = model.compileSpace()
        val projection = SearchTransform(space)
        val backend = WitnessSearch { _, _ ->
            com.eignex.combo.shared.HybridDescentSession.klause(space).witnesses(
                com.eignex.combo.shared.WitnessRequest(
                    1,
                    com.eignex.klause.propagation.Assumptions(
                        ints = mapOf(space.compiled.intVarIdByName.getValue(model.x.name) to 1L),
                    ),
                ),
                Random(1),
            )
        }
        val selected = LocalAndRandomSearch(projection, backend, localSteps = 12, neighbors = 32).maximize(
            Acquisition { rows -> DenseVector.of(DoubleArray(rows.rows) { rows[it, 0] }) },
            AcquisitionRequest(Context.Empty, emptyList(), emptyList()),
            Random(5),
        )
        assertEquals(128L, space.compiled.decode(model.x, selected.sample!!.sample))
        assertTrue(com.eignex.combo.shared.WitnessValidator(space).valid(selected.sample.sample))
    }

    private class Fixed : DecisionSpace() {
        val flag by boolVar()
        val fixed by constraint { !flag }
    }

    @Test
    fun `completed deterministic points are excluded while noisy or explicitly allowed replicates remain available`() {
        for ((noise, policy, repeats) in listOf(
            Triple(ObservationNoise.Deterministic, RepeatPolicy.NoisyOnly, false),
            Triple(ObservationNoise.Noisy, RepeatPolicy.NoisyOnly, true),
            Triple(ObservationNoise.Deterministic, RepeatPolicy.Allow, true),
            Triple(ObservationNoise.Noisy, RepeatPolicy.Never, false),
        )) {
            val optimizer = RandomForestOptimizer(
                SearchTransform(Fixed().compileSpace()),
                config = ForestOptimizerConfig(repeatPolicy = policy),
                outcome = OutcomeSpec(noise = noise),
            )
            val first = optimizer.suggest(1)[0]
            optimizer.complete(first.id, 2.0)
            val second = optimizer.suggest(1)
            assertEquals(if (repeats) 1 else 0, second.size)
            if (repeats) {
                assertEquals(first.sample, second[0].sample)
            } else {
                assertEquals(WitnessTermination.BudgetExhausted, second.termination)
            }
            assertEquals(1L, optimizer.store.revision)
        }
    }
}
