package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.decisions.FeatureEncoder
import com.eignex.combo.decisions.Scale
import com.eignex.combo.decisions.SubSpace
import com.eignex.combo.decisions.context
import com.eignex.combo.shared.ScaledWitnessProposals
import com.eignex.combo.shared.WitnessBatch
import com.eignex.combo.shared.WitnessSearch
import com.eignex.combo.shared.WitnessTermination
import com.eignex.combo.shared.WitnessValidator
import com.eignex.combo.test.TimedTest
import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.backtrack.BacktrackSolver
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.propagation.bake
import com.eignex.klause.schema.eq
import com.eignex.klause.schema.plus
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.Sample
import com.eignex.koblas.DenseVector
import com.eignex.koblas.Vector
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

class SearchTransformTest : TimedTest() {
    private class Child : SubSpace() {
        val rate by floatVar(1e-6, 1.0, scale = Scale.Log)
        val kind by nominal("a", "b")
    }
    private class Mixed : DecisionSpace() {
        val count by intVar(2, 10)
        val rate by floatVar(1e-6, 1.0, scale = Scale.Log)
        val decay by floatVar(0.0, 99.0, scale = Scale.ReverseLog)
        val finite by floatVar(-2.0, 2.0, buckets = 9)
        val mode by nominal("red", "blue")
        val child by optionalDecisionSpace(::Child)
        val level by optionalContextInt(1, 100, scale = Scale.Log)
        val premium by contextBool()
        val countEffect by interact(level, count)
        val modeEffect by interact(premium, mode)
    }

    private class Coupled : DecisionSpace() {
        val x by floatVar(0.0, 1.0)
        val y by floatVar(0.0, 1.0)
        val equality by constraint { (x + y) eq 1.0 }
    }

    private fun sample(space: com.eignex.combo.decisions.CompiledDecisionSpace, assumptions: Assumptions) =
        BacktrackSolver(space.compiled.problem.bake()).sample(
            BacktrackParams(maxDecisions = 64, assumptions = assumptions, randomSeed = 3),
        ).assignment!!

    @Test
    fun `unchanged coordinates preserve exact rational witnesses through decoding`() {
        val space = Coupled().compileSpace()
        val x = BigFraction.ofDouble(3.0)!!.reciprocal()
        val y = BigFraction.ONE - x
        val point = BanditSample(
            Sample(
                booleanArrayOf(),
                longArrayOf(),
                doubleArrayOf(x.toDouble(), y.toDouble()),
                exactReals = listOf(x, y),
            ),
        )
        val coordinates = SearchTransform(space)
        val decoded = coordinates.decode(coordinates.encode(point), point)!!
        assertEquals(point.sample.exactReals, decoded.sample.exactReals)
        assertTrue(WitnessValidator(space).valid(decoded.sample))
    }

    @Test
    fun `base coordinates round trip mixed variables without interaction columns`() {
        val model = Mixed()
        val space = model.compileSpace()
        val coordinates = SearchTransform(space)
        val features = InteractionFeatureProjection(coordinates)
        val ctx = context {
            set(model.level, 10)
            set(model.premium, true)
        }
        val samples = listOf(sample(space, space.assumptionsFor(ctx)))
        assertEquals(
            space.compiled.problem.let { it.numBoolVars + it.numIntVars + it.numRealVars },
            coordinates.featureSize,
        )
        assertEquals(coordinates.featureSize + 3, features.featureSize)
        for (raw in samples) {
            val sample = BanditSample(raw)
            val base = coordinates.encode(sample, ctx)
            assertTrue(base.toDoubleArray().all { it in 0.0..1.0 })
            val decoded = coordinates.decode(base, sample, ctx)!!
            assertEquals(sample.bools.toList(), decoded.bools.toList())
            assertEquals(sample.ints.toList(), decoded.ints.toList())
            assertEquals(sample.float(model.rate, space), decoded.float(model.rate, space), 1e-12)
            assertEquals(sample.float(model.decay, space), decoded.float(model.decay, space), 1e-12)
            val expanded = features.encode(sample, ctx)
            assertEquals(base.toDoubleArray().toList(), expanded.toDoubleArray().take(base.size))
            val countSlot = coordinates.layout.intStart + space.compiled.intVarIdByName.getValue(model.count.name)
            assertEquals(0.5 * base[countSlot], expanded[base.size], 1e-12)
            assertEquals(1.0, expanded[base.size + 1] + expanded[base.size + 2])
            assertFailsWith<IllegalArgumentException> { coordinates.decode(expanded, sample, ctx) }
        }
    }

    @Test
    fun `activity masks survive normalization and optional model expansion`() {
        val model = Mixed()
        val space = model.compileSpace()
        val coordinates = SearchTransform(space)
        val ctx = context { set(model.premium, true) }
        val gate = space.compiled.boolVarIdByName.getValue(space.gateOf(model.child)!!.name)
        val assumptions = space.assumptionsFor(ctx).mergedWith(Assumptions(bools = mapOf(gate to false)))
        val raw = sample(space, assumptions)
        val sample = BanditSample(raw)
        val base = coordinates.encode(sample, ctx)
        val levelSlot = coordinates.layout.intStart + space.compiled.intVarIdByName.getValue(model.level.name)
        assertEquals(0.0, base[levelSlot])
        assertEquals(0.0, InteractionFeatureProjection(coordinates).encode(sample, ctx)[base.size])
        val childId = coordinates.layout.realHandles.entries.single { it.value.name == model.child.rate.name }.key
        assertEquals(0.0, base[coordinates.layout.realStart + childId])
        for (id in space.compiled.nominalIndicators.getValue(model.child.kind.name).values) assertEquals(0.0, base[id])
        assertTrue(coordinates.decode(base, sample, ctx) != null)
        val proposals = ScaledWitnessProposals(space)
        val varied = List(4) { proposals.interior(raw, Random(it)) }
        assertTrue(varied.map { BanditSample(it).float(model.rate, space) }.distinct().size > 1)
        assertTrue(varied.all(WitnessValidator(space)::valid))
    }

    @Test
    fun `changing model feature expansion preserves feasible proposal geometry`() {
        val model = Mixed()
        val space = model.compileSpace()
        val coordinates = SearchTransform(space)
        val expanded = InteractionFeatureProjection(coordinates)
        val ctx = context {
            set(model.level, 10)
            set(model.premium, true)
        }
        val countSlot = coordinates.layout.intStart + space.compiled.intVarIdByName.getValue(model.count.name)
        fun select(projection: FeatureEncoder<Vector>, slot: Int): BanditSample {
            val seeds = listOf(2L, 10L).map { count ->
                sample(
                    space,
                    space.assumptionsFor(ctx).mergedWith(
                        Assumptions(
                            ints = mapOf(space.compiled.intVarIdByName.getValue(model.count.name) to count),
                        ),
                    ),
                )
            }
            val search = LocalAndRandomSearch(
                projection,
                WitnessSearch { _, _ -> WitnessBatch(seeds, WitnessTermination.CountReached) },
                localSteps = 0,
            )
            return search.maximize(
                Acquisition { rows -> DenseVector.of(DoubleArray(rows.rows) { rows[it, slot] }) },
                AcquisitionRequest(ctx, emptyList(), emptyList()),
                Random(17),
            ).sample!!
        }
        val basePoint = select(coordinates, countSlot)
        val expandedPoint = select(expanded, coordinates.featureSize)
        assertEquals(basePoint, expandedPoint)
        assertEquals(10L, space.compiled.decode(model.count, expandedPoint.sample))
    }

    @Test
    fun `decode rejects invalid categorical combinations and changed hard contexts`() {
        val model = Mixed()
        val space = model.compileSpace()
        val coordinates = SearchTransform(space)
        val ctx = context { set(model.premium, true) }
        val sample = BanditSample(
            sample(space, space.assumptionsFor(ctx)),
        )
        val invalidCategory = DenseVector.of(coordinates.encode(sample, ctx).toDoubleArray())
        for (id in space.compiled.nominalIndicators.getValue(model.mode.name).values) invalidCategory[id] = 1.0
        assertNull(coordinates.decode(invalidCategory, sample, ctx))
        val invalidContext = DenseVector.of(coordinates.encode(sample, ctx).toDoubleArray())
        invalidContext[space.compiled.boolVarIdByName.getValue(model.premium.name)] = 0.0
        assertNull(coordinates.decode(invalidContext, sample, ctx))
    }
}
