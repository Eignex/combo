package combo.bandit.glm

import com.eignex.klause.solver.localsearch.LocalSearchParams
import com.eignex.klause.solver.localsearch.LocalSearchSolver
import combo.decisions.Context
import combo.decisions.DecisionSpace
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Round-trip tests for the [LinearFeatureLayout] float-scaling translation.
 *
 * The bandit fits its weight `w` in per-unit-real-value Double units. Klause solves
 * `LinearObjective` over bucketed integers. The contract:
 *
 *   model_weight · real_value(bucket)
 *      ≡  bucket_coefficient · bucket + constant_contribution
 *
 * Verifying this exactly for every (handle, bucket) pair pins down the scaling math.
 */
private class FloatHostSpace : DecisionSpace() {
    val temp by floatVar(min = 0.0, max = 100.0, buckets = 11)    // scale = 10,  offset = 0
    val pressure by floatVar(min = -5.0, max = 5.0, buckets = 21) // scale = 0.5, offset = -5
    val count by intVar(0, 100)                                   // true int, identity
}

class LinearFeatureFloatScalingTest {

    @Test
    fun `realValue is identity for true ints, decoded Double for floats`() {
        val space = FloatHostSpace().compileSpace()
        val layout = LinearFeatureLayout.from(space)

        val tempId = space.compiled.intVarIdByName.getValue("temp")
        val pressureId = space.compiled.intVarIdByName.getValue("pressure")
        val countId = space.compiled.intVarIdByName.getValue("count")

        assertEquals(42.0, layout.realValue(countId, 42))
        assertEquals(0.0, layout.realValue(tempId, 0))
        assertEquals(100.0, layout.realValue(tempId, 10))
        assertEquals(50.0, layout.realValue(tempId, 5))
        assertEquals(-5.0, layout.realValue(pressureId, 0))
        assertEquals(0.0, layout.realValue(pressureId, 10))
        assertEquals(5.0, layout.realValue(pressureId, 20))
    }

    @Test
    fun `coefficientForInt rewrites model weight to bucket units with constant offset`() {
        val space = FloatHostSpace().compileSpace()
        val layout = LinearFeatureLayout.from(space)
        val tempId = space.compiled.intVarIdByName.getValue("temp")
        val countId = space.compiled.intVarIdByName.getValue("count")

        val (coeffCount, constCount) = layout.coefficientForInt(countId, 2.5)
        assertEquals(2.5, coeffCount)
        assertEquals(0.0, constCount)

        // Temp: scale = 10, offset = 0. Per-real weight 0.3 → 3.0 per bucket.
        val (coeffTemp, constTemp) = layout.coefficientForInt(tempId, 0.3)
        assertEquals(3.0, coeffTemp, 1e-12)
        assertEquals(0.0, constTemp, 1e-12)
    }

    @Test
    fun `coefficient round-trip equals model_weight times real_value for every bucket`() {
        val space = FloatHostSpace().compileSpace()
        val layout = LinearFeatureLayout.from(space)
        val pressureId = space.compiled.intVarIdByName.getValue("pressure")
        val w = -1.7  // arbitrary per-real-unit weight, negative for sign check

        val (bucketCoeff, constantContribution) = layout.coefficientForInt(pressureId, w)
        for (bucket in 0..20) {
            val real = layout.realValue(pressureId, bucket)
            val modelSide = w * real
            val klauseSide = bucketCoeff * bucket + constantContribution
            assertEquals(
                modelSide, klauseSide, 1e-9,
                "round-trip mismatch at bucket=$bucket: model=$modelSide klause=$klauseSide",
            )
        }
    }

    @Test
    fun `encode emits real-value features at float slots`() {
        val model = FloatHostSpace()
        val space = model.compileSpace()
        val projection = LinearFeatureProjection(space)
        val tempId = space.compiled.intVarIdByName.getValue("temp")
        val pressureId = space.compiled.intVarIdByName.getValue("pressure")
        val countId = space.compiled.intVarIdByName.getValue("count")

        val solver = LocalSearchSolver(space.compiled.problem)
        val sample = solver.sample(LocalSearchParams(randomSeed = 1L))!!
        // Undithered wrap on purpose: this test verifies the bucket-midpoint encoding,
        // which is what features look like in the absence of dither.
        val features = projection.encode(combo.decisions.BanditSample.undithered(sample), Context.Empty)

        val tempReal = projection.layout.realValue(tempId, sample.ints[tempId])
        val pressureReal = projection.layout.realValue(pressureId, sample.ints[pressureId])
        assertEquals(tempReal.toFloat(), features[projection.layout.intStart + tempId])
        assertEquals(pressureReal.toFloat(), features[projection.layout.intStart + pressureId])
        assertEquals(sample.ints[countId].toFloat(), features[projection.layout.intStart + countId])

        assertTrue(tempReal in 0.0..100.0)
        assertTrue(pressureReal in -5.0..5.0)
    }
}
