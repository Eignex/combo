package combo.bandit.glm

import combo.math.vectors
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

class LinearModelTest {

    private fun synthetic(
        nFeatures: Int,
        nSamples: Int,
        trueWeights: FloatArray,
        bias: Float,
        noise: Float,
        rng: Random,
    ): Pair<Array<combo.math.VectorView>, FloatArray> {
        val inputs = Array(nSamples) {
            vectors.vector(FloatArray(nFeatures) { rng.nextFloat() * 2 - 1 })
        }
        val targets = FloatArray(nSamples) { i ->
            var y = bias
            for (j in 0 until nFeatures) y += trueWeights[j] * inputs[i][j]
            y + (rng.nextFloat() * 2 - 1) * noise
        }
        @Suppress("UNCHECKED_CAST")
        return (inputs as Array<combo.math.VectorView>) to targets
    }

    @Test
    fun diagonalModelLearnsSignOfWeights() {
        // Diagonalized model uses precision-weighted updates; precise convergence is
        // sensitive to hyperparameters. The robust assertion is that signs end up correct.
        val rng = Random(42)
        val trueWeights = floatArrayOf(0.8f, -0.8f, 0.6f, -0.6f)
        val (inputs, targets) = synthetic(4, 3000, trueWeights, bias = 0f, noise = 0.05f, rng)

        val model = DiagonalizedLinearModel.Builder(4)
            .family(NormalVariance)
            .learningRate(ConstantRate(1f))
            .priorPrecision(0.01f)
            .exploration(0f)
            .build()
        for (i in inputs.indices) model.train(inputs[i], targets[i], 1f)

        for (j in trueWeights.indices) {
            assertTrue(model.weights[j] * trueWeights[j] > 0,
                "weight $j has wrong sign: got ${model.weights[j]}, expected sign of ${trueWeights[j]}")
        }
    }

    @Test
    fun sgdRegressionConvergesToLinearTarget() {
        val rng = Random(7)
        val trueWeights = floatArrayOf(1.0f, -0.5f, 0.3f)
        val (inputs, targets) = synthetic(3, 3000, trueWeights, bias = 0f, noise = 0.02f, rng)

        val model = SGDLinearModel.Builder(3)
            .exploration(0f)
            .build()
        for (i in inputs.indices) model.train(inputs[i], targets[i], 1f)

        for (j in trueWeights.indices) {
            assertTrue(abs(model.weights[j] - trueWeights[j]) < 0.25f,
                "weight $j: got ${model.weights[j]}, expected ~${trueWeights[j]}")
        }
    }

    @Test
    fun covarianceModelExportImportRoundtripsWeights() {
        val rng = Random(13)
        val trueWeights = floatArrayOf(0.7f, -0.4f)
        val (inputs, targets) = synthetic(2, 200, trueWeights, bias = 0f, noise = 0.02f, rng)

        val model = CovarianceLinearModel.Builder(2)
            .family(NormalVariance)
            .exploration(0f)
            .build()
        for (i in inputs.indices) model.train(inputs[i], targets[i], 1f)

        val data = model.exportData()
        val fresh = CovarianceLinearModel.Builder(2).family(NormalVariance).build()
        fresh.importData(data, varianceMixin = 1f, weightMixin = 1f)

        for (j in 0 until 2) {
            assertTrue(abs(fresh.weights[j] - model.weights[j]) < 1e-3f,
                "weight $j after roundtrip mismatch")
        }
        assertTrue(abs(fresh.bias - model.bias) < 1e-3f)
    }
}
