package combo.math

import combo.sat.BitArray
import combo.test.assertContentEquals
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertContentEquals

class CholeskyTest {
    @Test
    fun choleskyTest() {
        val H = FloatMatrix(arrayOf(
                floatArrayOf(1.0565f, 0.3456f, -0.4646f, 1.8587f),
                floatArrayOf(0.3456f, 0.5910f, -0.1395f, 0.6277f),
                floatArrayOf(-0.4646f, -0.1395f, 0.2371f, -0.8419f),
                floatArrayOf(1.8587f, 0.6277f, -0.8419f, 4.2134f)))

        val L = arrayOf(
                floatArrayOf(1.0279f, 0.3362f, -0.4520f, 1.8083f),
                floatArrayOf(0.0f, 0.6913f, 0.0181f, 0.0285f),
                floatArrayOf(0.0f, 0.0f, 0.1799f, -0.1392f),
                floatArrayOf(0.0f, 0.0f, 0.0f, 0.9608f))

        val cholesky = H.cholesky().T
        assertContentEquals(L, cholesky.toArray(), 1e-3f)
    }

    @Test
    fun choleskyDowndate() {
        val Hinv = FloatMatrix(arrayOf(
                floatArrayOf(1.0565f, 0.3456f, -0.4646f, 1.8587f),
                floatArrayOf(0.3456f, 0.5910f, -0.1395f, 0.6277f),
                floatArrayOf(-0.4646f, -0.1395f, 0.2371f, -0.8419f),
                floatArrayOf(1.8587f, 0.6277f, -0.8419f, 4.2134f)))

        val x = FloatVector(floatArrayOf(1f, 0f, 1f, 1f))
        val s = sqrt(2f)
        val u = x / s
        val L = Hinv.cholesky().T
        val z = (Hinv * u) / sqrt(1f + (u dot Hinv * u))
        L.choleskyDowndate(z)

        val expected = arrayOf(
                floatArrayOf(0.5993f, 0.1808f, -0.2675f, 0.6179f),
                floatArrayOf(0f, 0.6911f, 0.0180f, 0.0139f),
                floatArrayOf(0f, 0f, 0.1801f, -0.1520f),
                floatArrayOf(0f, 0f, 0f, 0.7948f))

        assertContentEquals(expected, L.toArray(), 1e-4f)
    }

    @Test
    fun choleskyDowndate2() {
        val Hinv = FloatMatrix(arrayOf(
                floatArrayOf(28.6684f, 104.7670f, -129.7711f, -15.8332f),
                floatArrayOf(104.7670f, 392.5219f, -484.8603f, -60.7211f),
                floatArrayOf(-129.7711f, -484.8603f, 599.5189f, 74.8766f),
                floatArrayOf(-15.8332f, -60.7211f, 74.8766f, 9.8561f)))

        val x = BitArray(4)
        x[0] = true; x[3] = true
        val s = 2f
        val u = x / s
        val L = Hinv.cholesky().T
        val z = (Hinv * u) / sqrt(1f + (u dot Hinv * u))
        L.choleskyDowndate(z)

        val expected = arrayOf(
                floatArrayOf(3.6737f, 14.3455f, -17.6610f, -2.3866f),
                floatArrayOf(0f, 2.8386f, -3.1094f, -0.7884f),
                floatArrayOf(0f, 0f, 0.6445f, 0.0889f),
                floatArrayOf(0f, 0f, 0f, 0.4904f))

        assertContentEquals(expected, L.toArray(), 1e-4f)
    }
}
