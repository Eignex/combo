package com.eignex.combo.bo

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ForestUncertaintyTest {
    @Test
    fun `leaf mean uncertainty propagates supplied noise while disagreement remains unchanged`() {
        val query = DenseMatrix.ofRows(arrayOf(doubleArrayOf(0.0)))
        val data = TrainingData(
            query,
            DenseVector.of(doubleArrayOf(2.0)),
            DenseVector.of(doubleArrayOf(1.0)),
            DenseVector.of(doubleArrayOf(4.0)),
        )
        val original = ForestSurrogate().fit(data, Random(3)).predict(query)
        val estimated = ForestSurrogate(uncertainty = ForestUncertainty.LeafMean).fit(data, Random(3))
        assertEquals(0.0, original.variances[0])
        assertEquals(4.0, estimated.predict(query).variances[0])
        assertEquals(original.means[0], estimated.predict(query).means[0])
        assertEquals(UncertaintyKind.EstimatedMean, estimated.uncertaintyKind)
        val draw = estimated.drawFunction(Random(2))
        assertEquals(draw.evaluate(query)[0], draw.evaluate(query)[0])
        assertTrue(draw.evaluate(query)[0] != 2.0)
    }
}
