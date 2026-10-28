package com.eignex.combo.bandit.glm

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.shared.LinearFeatureLayout
import com.eignex.combo.shared.LinearFeatureProjection
import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.backtrack.BacktrackSolver
import com.eignex.klause.model.FloatSpec
import com.eignex.klause.propagation.bake
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class LinearFeatureProjectionTest {
    private class Space : DecisionSpace() {
        val temperature by floatVar(0.0, 100.0)
        val pressure by floatVar(-5.0, 5.0)
        val count by intVar(0, 100)
    }

    @Test
    fun `float declarations retain Klause defaults without a Combo grid`() {
        val space = Space().compileSpace()
        for (spec in space.definition.variables.values.filterIsInstance<FloatSpec>()) {
            assertNull(spec.buckets)
        }
        val layout = LinearFeatureLayout.from(space)
        assertEquals(1, layout.numIntVars)
        assertEquals(2, layout.numRealVars)
        assertEquals(3, layout.featureSize)
    }

    @Test
    fun `encoded float features equal the values Klause decodes`() {
        val model = Space()
        val space = model.compileSpace()
        val projection = LinearFeatureProjection(space)
        val raw = BacktrackSolver(space.compiled.problem.bake()).sample(BacktrackParams(randomSeed = 1L)).assignment!!
        val sample = BanditSample(raw)
        val features = projection.encode(sample, Context.Empty)
        for ((id, handle) in projection.layout.realHandles) {
            assertEquals(space.compiled.decode(handle, raw), features[projection.layout.realStart + id])
            assertEquals(space.compiled.decode(handle, raw), sample.float(handle, space))
        }
        val countId = space.compiled.intVarIdByName.getValue(model.count.name)
        assertEquals(space.compiled.decode(model.count, raw).toDouble(), features[projection.layout.intStart + countId])
    }
}
