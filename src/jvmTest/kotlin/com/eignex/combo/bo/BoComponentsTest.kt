package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.decisions.FeatureEncoder
import com.eignex.combo.decisions.Scale
import com.eignex.combo.decisions.context
import com.eignex.combo.shared.WitnessBatch
import com.eignex.combo.shared.WitnessSearch
import com.eignex.combo.shared.WitnessTermination
import com.eignex.klause.schema.eq
import com.eignex.klause.schema.le
import com.eignex.klause.schema.plus
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.Sample
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.Vector
import com.eignex.kumulant.stat.regression.tree.BatchForestConfig
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class BoComponentsTest {
    private class Rate : DecisionSpace() {
        val value by floatVar(0.0, 1.0)
    }
    private fun point(value: Double) = BanditSample(
        Sample(
            booleanArrayOf(),
            longArrayOf(),
            doubleArrayOf(value),
            exactReals = listOf(BigFraction.ofDouble(value)!!),
        ),
    )

    @Test
    fun `coherent forest function draws preserve repeated and reordered predictions`() {
        val data = TrainingData(
            DenseMatrix.ofRows(arrayOf(doubleArrayOf(0.0), doubleArrayOf(1.0))),
            DenseVector.of(doubleArrayOf(0.0, 9.0)),
            DenseVector.of(doubleArrayOf(1.0, 1.0)),
        )
        val fitted = ForestSurrogate(BatchForestConfig(nbrTrees = 5)).fit(data, Random(3))
        val queries = DenseMatrix.ofRows(arrayOf(doubleArrayOf(0.0), doubleArrayOf(1.0), doubleArrayOf(0.0)))
        val draw = fitted.drawFunction(Random(2))
        val first = draw.evaluate(queries)
        assertEquals(first[0], first[2])
        assertEquals(first.toDoubleArray().toList(), draw.evaluate(queries).toDoubleArray().toList())
        data.features[0, 0] = 1000.0
        assertEquals(first.toDoubleArray().toList(), draw.evaluate(queries).toDoubleArray().toList())
        assertEquals(UncertaintyKind.EnsembleDisagreement, fitted.uncertaintyKind)
    }

    @Test
    fun `acquisition guided local search improves beyond the initial witness pool`() {
        val space = Rate().compileSpace()
        val projection = SearchTransform(space)
        val onlyZero = WitnessSearch { _, _ ->
            WitnessBatch(
                listOf(point(0.0).sample),
                WitnessTermination.CountReached,
            )
        }
        val objective = Acquisition { features ->
            DenseVector.of(
                DoubleArray(features.rows) {
                    -(features[it, 0] - 0.73) * (features[it, 0] - 0.73)
                },
            )
        }
        val request = AcquisitionRequest(Context.Empty, emptyList(), emptyList())
        val initial = LocalAndRandomSearch(projection, onlyZero, candidateCount = 1, localSteps = 0)
            .maximize(objective, request, Random(7))
        val refined = LocalAndRandomSearch(projection, onlyZero, candidateCount = 1, localSteps = 30, neighbors = 20)
            .maximize(objective, request, Random(7))
        assertEquals(0.0, initial.sample!!.sample.reals[0])
        assertEquals(0.73, refined.sample!!.sample.reals[0], 0.02)
        assertTrue(refined.evaluations > initial.evaluations)
    }

    @Test
    fun `acquisition batches capture rows from reusable encoder storage`() {
        val compiled = Rate().compileSpace()
        val encoder = object : FeatureEncoder<Vector> {
            override val space = compiled
            override val featureSize = 1
            private val buffer = DenseVector.zero(1)
            override fun encode(sample: BanditSample, context: Context): Vector {
                buffer[0] = sample.sample.reals[0]
                return buffer
            }
        }
        val backend = WitnessSearch { _, _ ->
            WitnessBatch(listOf(point(1.0).sample, point(0.0).sample), WitnessTermination.CountReached)
        }
        val selected = LocalAndRandomSearch(encoder, backend, candidateCount = 2, localSteps = 0)
            .maximize(
                Acquisition { rows -> DenseVector.of(DoubleArray(rows.rows) { rows[it, 0] }) },
                AcquisitionRequest(Context.Empty, emptyList(), emptyList()),
                Random(1),
            )
        assertEquals(1.0, selected.sample!!.sample.reals[0])
    }

    private class Capped : DecisionSpace() {
        val value by floatVar(0.0, 1.0)
        val premium by contextBool()
        val cap by constraint { value le 0.4 }
    }

    @Test
    fun `local acquisition refinement retains original constraints and context`() {
        val model = Capped()
        val space = model.compileSpace()
        val projection = SearchTransform(space)
        val ctx = context { set(model.premium, true) }
        val anchor = Sample(
            booleanArrayOf(true),
            longArrayOf(),
            doubleArrayOf(0.0),
            exactReals = listOf(BigFraction.ZERO),
        )
        val backend = WitnessSearch { _, _ -> WitnessBatch(listOf(anchor), WitnessTermination.CountReached) }
        val selected = LocalAndRandomSearch(projection, backend, candidateCount = 1, localSteps = 12)
            .maximize(
                Acquisition { x -> DenseVector.of(DoubleArray(x.rows) { x[it, 1] }) },
                AcquisitionRequest(ctx, emptyList(), emptyList()),
                Random(4),
            )
        assertTrue(space.matches(selected.sample!!.sample, space.assumptionsFor(ctx)))
        assertTrue(selected.sample!!.float(model.value, space) in 0.3..0.4)
    }

    private class Coupled : DecisionSpace() {
        val x by floatVar(0.0, 1.0)
        val y by floatVar(0.0, 1.0)
        val equality by constraint { (x + y) eq 1.0 }
    }

    @Test
    fun `joint real moves preserve exact affine equalities and reach feasible interiors`() {
        val model = Coupled()
        val space = model.compileSpace()
        fun vertex(x: Double, y: Double) = Sample(
            booleanArrayOf(),
            longArrayOf(),
            doubleArrayOf(x, y),
            exactReals = listOf(BigFraction.ofDouble(x)!!, BigFraction.ofDouble(y)!!),
        )
        val seeds = listOf(vertex(0.0, 1.0), vertex(1.0, 0.0))
        val backend = WitnessSearch { _, _ -> WitnessBatch(seeds, WitnessTermination.CountReached) }
        val selected = LocalAndRandomSearch(SearchTransform(space), backend, candidateCount = 2, localSteps = 2)
            .maximize(
                Acquisition { rows ->
                    DenseVector.of(
                        DoubleArray(rows.rows) {
                            -(rows[it, 0] - 0.5) * (rows[it, 0] - 0.5)
                        },
                    )
                },
                AcquisitionRequest(Context.Empty, emptyList(), emptyList()),
                Random(1),
            ).sample!!
        assertEquals(0.5, selected.float(model.x, space), 1e-12)
        assertEquals(0.5, selected.float(model.y, space), 1e-12)
        val exact = selected.sample.exactReals!!
        assertEquals(BigFraction.ONE, exact[0] + exact[1])
    }

    @Test
    fun `study rejects stale foreign and duplicate reservations without partial commits`() {
        val space = Rate().compileSpace()
        val study = Study(space)
        val snapshot = study.beginRound()
        study.store.append(Observation(point(0.1), Context.Empty, 1.0))
        assertFailsWith<IllegalStateException> {
            study.reserve(snapshot, ProposalBatch(listOf(point(0.2)), WitnessTermination.CountReached))
        }
        val current = study.beginRound()
        assertFailsWith<IllegalArgumentException> {
            study.reserve(current, ProposalBatch(listOf(point(0.2), point(0.2)), WitnessTermination.CountReached))
        }
        assertTrue(study.trials.isEmpty())
        assertFailsWith<IllegalStateException> {
            study.reserve(Study(space).beginRound(), ProposalBatch(listOf(point(0.2)), WitnessTermination.CountReached))
        }
    }

    private class LogRate : DecisionSpace() {
        val value by floatVar(1e-6, 1.0, scale = Scale.Log)
    }

    @Test
    fun `BO coordinates normalize declared logarithmic geometry without changing original assignments`() {
        val model = LogRate()
        val space = model.compileSpace()
        val transform = SearchTransform(space)
        assertEquals(0.0, transform.encode(point(1e-6))[0], 1e-12)
        assertEquals(0.5, transform.encode(point(1e-3))[0], 1e-12)
        assertEquals(1.0, transform.encode(point(1.0))[0], 1e-12)
        assertEquals(1e-3, point(1e-3).float(model.value, space))
        val decoded = transform.decode(transform.encode(point(1e-3)), point(1e-6))!!
        assertEquals(1e-3, decoded.float(model.value, space), 1e-12)
    }

    @Test
    fun `noise aware observed recommendations aggregate replicates independently of known variance`() {
        val space = Rate().compileSpace()
        val noisy = RandomForestOptimizer(SearchTransform(space), outcome = OutcomeSpec(noise = ObservationNoise.Noisy))
        noisy.train(point(0.0), Context.Empty, 100.0, weight = 1.0, noiseVariance = 25.0)
        noisy.train(point(0.0), Context.Empty, 0.0, weight = 49.0, noiseVariance = 1.0)
        noisy.train(point(1.0), Context.Empty, 5.0, weight = 2.0)
        assertEquals(point(1.0), noisy.bestObserved()!!.sample)
        val deterministic = RandomForestOptimizer.smac(space)
        deterministic.importData(noisy.exportData())
        assertEquals(point(0.0), deterministic.bestObserved()!!.sample)
        assertEquals(25.0, noisy.store.snapshot().observations.first().noiseVariance)
    }

    @Test
    fun `compatible study transfer rebinds contextual handles to the new compiled schema`() {
        val first = Capped()
        val left = RandomForestOptimizer.smac(first.compileSpace())
        val trial = left.suggest(1, context { set(first.premium, true) })[0]
        val second = Capped()
        val right = RandomForestOptimizer.smac(second.compileSpace())
        right.importData(left.exportData())
        right.complete(trial.id, 2.0)
        val ctx = context { set(second.premium, true) }
        assertEquals(2.0, right.predict(trial.sample, ctx))
        assertEquals(1L, right.store.revision)
    }

    @Test
    fun `response standardization returns model predictions in original reward units`() {
        val space = Rate().compileSpace()
        val optimizer = RandomForestOptimizer(
            SearchTransform(space),
            config = ForestOptimizerConfig(forest = BatchForestConfig(nbrTrees = 1, bootstrap = false)),
            outcome = OutcomeSpec(standardize = true),
        )
        optimizer.train(point(0.0), 100.0)
        optimizer.train(point(1.0), 300.0)
        assertEquals(100.0, optimizer.predict(point(0.0)), 1e-10)
        assertEquals(300.0, optimizer.predict(point(1.0)), 1e-10)
    }
}
