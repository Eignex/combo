package combo.math

import combo.test.assertContentEquals
import kotlin.math.absoluteValue
import kotlin.math.roundToInt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

abstract class VectorTest(val vectorFactory: VectorFactory) {

    fun vector(vararg xs: Float): Vector = vectorFactory.vector(xs)
    fun zeroVector(size: Int): Vector = vectorFactory.zeroVector(size)

    @Test
    fun size() {
        assertEquals(0, vector().size)
        assertEquals(3, vector(1f, 2f, 3f).size)
        assertEquals(2, vector(-1f, 2f).size)
    }

    @Test
    fun zeroVectorSize() {
        for (i in 0 until 10) {
            assertEquals(i, zeroVector(i).size)
        }
    }

    @Test
    fun dotProduct() {
        val v = vector(-1.0f, 2.0f, 0.0f)
        val u = vector(0.0f, 3.0f, 4.0f)
        assertEquals(6.0f, u dot v)
        assertEquals(6.0f, v dot u)
        assertContentEquals(floatArrayOf(-1.0f, 2.0f, 0.0f), v.toFloatArray())
        assertContentEquals(floatArrayOf(0.0f, 3.0f, 4.0f), u.toFloatArray())
    }

    @Test
    fun dotProductZero() {
        val v = vector(-1.0f, 2.0f, 0.0f)
        val u = zeroVector(3)
        assertEquals(0.0f, u dot v)
        assertEquals(0.0f, v dot u)
    }

    @Test
    fun dotProductFallback() {
        val v = vector(-1.0f, 2.0f, 0.0f)
        val u = FloatVector(floatArrayOf(0.0f, 3.0f, 4.0f))
        assertEquals(6.0f, u dot v)
        assertEquals(6.0f, v dot u)
        assertContentEquals(floatArrayOf(-1.0f, 2.0f, 0.0f), v.toFloatArray())
        assertContentEquals(floatArrayOf(0.0f, 3.0f, 4.0f), u.toFloatArray())
    }

    @Test
    fun norm2() {
        val v = vector(-1.0f, 2.0f, 0.0f, -2.0f)
        assertEquals(3.0f, v.norm2())
        assertContentEquals(floatArrayOf(-1.0f, 2.0f, 0.0f, -2.0f), v.toFloatArray())
    }

    @Test
    fun plusScalar() {
        val v = vector(-1.0f, 2.0f, 0.0f) + 2f
        assertContentEquals(floatArrayOf(1.0f, 4.0f, 2.0f), v.toFloatArray())
    }

    @Test
    fun plusVector() {
        val v1 = vector(-1.0f, 2.0f, 0.0f)
        val v2 = vector(-2.0f, 0f, 4f)
        val u1 = v1 + v2
        val u2 = v2 + v1
        assertContentEquals(floatArrayOf(-3.0f, 2.0f, 4.0f), u1.toFloatArray())
        assertContentEquals(floatArrayOf(-3.0f, 2.0f, 4.0f), u2.toFloatArray())
    }

    @Test
    fun minusScalar() {
        val v = vector(-1.0f, 2.0f, 0.0f) - 2.0f
        assertContentEquals(floatArrayOf(-3.0f, 0.0f, -2.0f), v.toFloatArray())
    }

    @Test
    fun minusVector() {
        val v1 = vector(-1.0f, 2.0f, 0.0f)
        val v2 = vector(-2.0f, 0f, 4f)
        val u1 = v1 - v2
        val u2 = v2 - v1
        assertContentEquals(floatArrayOf(1.0f, 2.0f, -4.0f), u1.toFloatArray())
        assertContentEquals(floatArrayOf(-1.0f, -2.0f, 4.0f), u2.toFloatArray())
    }

    @Test
    fun timesScalar() {
        val v = vector(-1.0f, 2.0f, 0.0f) * 2f
        assertContentEquals(floatArrayOf(-2.0f, 4.0f, 0.0f), v.toFloatArray())
    }

    @Test
    fun timesVector() {
        val v1 = vector(-1.0f, 2.0f, 0.0f)
        val v2 = vector(-2.0f, 0f, 4f)
        val u1 = v1 * v2
        val u2 = v2 * v1
        assertContentEquals(floatArrayOf(2.0f, 0.0f, 0.0f), u1.toFloatArray())
        assertContentEquals(floatArrayOf(2.0f, 0.0f, 0.0f), u2.toFloatArray())
        assertContentEquals(floatArrayOf(-1.0f, 2.0f, 0.0f), v1.toFloatArray())
        assertContentEquals(floatArrayOf(-2.0f, 0.0f, 4.0f), v2.toFloatArray())
    }

    @Test
    fun divScalar() {
        val v = vector(-4.0f, 2.0f, 0.0f) / 2.0f
        assertContentEquals(floatArrayOf(-2.0f, 1.0f, 0.0f), v.toFloatArray())
    }

    @Test
    fun divVector() {
        val v1 = vector(-1.0f, 2.0f, 6.0f)
        val v2 = vector(-1.0f, 4.0f, 3.0f)
        val u1 = v1 / v2
        val u2 = v2 / v1
        assertContentEquals(floatArrayOf(1.0f, 0.5f, 2.0f), u1.toFloatArray())
        assertContentEquals(floatArrayOf(1.0f, 2.0f, 0.5f), u2.toFloatArray())
    }

    @Test
    fun toFloatArray() {
        val vector = vector(-1.2f, 2.1f, 1041.1f)
        assertContentEquals(floatArrayOf(-1.2f, 2.1f, 1041.1f), vector.toFloatArray())
    }

    @Test
    fun toIntArray() {
        val vector = vector(-1.2f, 2.1f, 1041.1f)
        assertContentEquals(intArrayOf(-1, 2, 1041), vector.toIntArray())
    }

    @Test
    fun toIntArrayGcd() {
        val rng = Random
        val vector = vector(*FloatArray(100) { rng.nextNormal() })
        val rounded = vector.toIntArray(1.0f, false)
        for (i in rounded.indices)
            assertEquals(rounded[i], vector[i].roundToInt())
        val larger = vector.toIntArray(1e-4f, false)
        for (i in rounded.indices) {
            assertTrue(larger[i].absoluteValue >= rounded[i].absoluteValue)
        }
    }

    @Test
    fun inlineAddScalar() {
        val v = vector(-1.0f, 2.0f, 0.0f)
        v.add(5.0f)
        assertContentEquals(floatArrayOf(4.0f, 7.0f, 5.0f), v.toFloatArray())
    }

    @Test
    fun inlineAddVector() {
        val v1 = vector(-1.0f, 2.0f, 0.0f)
        val v2 = vector(-2.0f, 0f, 4f)
        v1.add(v2)
        assertContentEquals(floatArrayOf(-3.0f, 2.0f, 4.0f), v1.toFloatArray())
        assertContentEquals(floatArrayOf(-2.0f, 0.0f, 4.0f), v2.toFloatArray())
    }

    @Test
    fun inlineSubScalar() {
        val v = vector(-1.0f, 2.0f, 0.0f)
        v.subtract(5.0f)
        assertContentEquals(floatArrayOf(-6.0f, -3.0f, -5.0f), v.toFloatArray())
    }

    @Test
    fun inlineSubVector() {
        val v1 = vector(-1.0f, 2.0f, 0.0f)
        val v2 = vector(-2.0f, 0f, 4f)
        v1.subtract(v2)
        assertContentEquals(floatArrayOf(1.0f, 2.0f, -4.0f), v1.toFloatArray())
        assertContentEquals(floatArrayOf(-2.0f, 0.0f, 4.0f), v2.toFloatArray())
    }

    @Test
    fun inlineMultiplyScalar() {
        val v = vector(-1.0f, 2.0f, 0.0f)
        v.multiply(5.0f)
        assertContentEquals(floatArrayOf(-5.0f, 10.0f, 0.0f), v.toFloatArray())
    }

    @Test
    fun inlineMultiplyVector() {
        val v1 = vector(-1.0f, 2.0f, 0.0f)
        val v2 = vector(-2.0f, 0f, 4f)
        v1.multiply(v2)
        assertContentEquals(floatArrayOf(2.0f, 0.0f, 0.0f), v1.toFloatArray())
        assertContentEquals(floatArrayOf(-2.0f, 0.0f, 4.0f), v2.toFloatArray())
    }

    @Test
    fun inlineDivideScalar() {
        val v = vector(-2.0f, 1.0f, 4.0f)
        v.divide(2.0f)
        assertContentEquals(floatArrayOf(-1.0f, 0.5f, 2.0f), v.toFloatArray())
    }

    @Test
    fun inlineDivideVector() {
        val v1 = vector(-1.0f, 2.0f, 0.0f)
        val v2 = vector(-2.0f, 1.0f, 4.0f)
        v1.divide(v2)
        assertContentEquals(floatArrayOf(0.5f, 2.0f, 0.0f), v1.toFloatArray())
        assertContentEquals(floatArrayOf(-2.0f, 1.0f, 4.0f), v2.toFloatArray())
    }
}

abstract class MatrixTest(val vectorFactory: VectorFactory) {

    fun zeroMatrix(rows: Int, cols: Int = rows): Matrix = vectorFactory.zeroMatrix(rows, cols)

    @Suppress("UNCHECKED_CAST")
    fun matrix(vararg xs: FloatArray): Matrix = vectorFactory.matrix(xs as Array<FloatArray>)

    fun vector(vararg xs: Float): Vector = vectorFactory.vector(xs)

    @Test
    fun emptyMatrix() {
        assertEquals(0, zeroMatrix(0).rows)
    }

    @Test
    fun matrixSize() {
        assertEquals(2, zeroMatrix(2).rows)
        assertEquals(2, zeroMatrix(2).cols)
        assertEquals(2, zeroMatrix(2, 3).rows)
        assertEquals(3, zeroMatrix(2, 3).cols)
    }

    @Test
    fun transpose() {
        val A = matrix(floatArrayOf(1.0f, 2.0f, 3.0f),
                floatArrayOf(4.0f, 5.0f, 6.0f),
                floatArrayOf(7.0f, 8.0f, 9.0f))

        val expected = arrayOf(
                floatArrayOf(1.0f, 4.0f, 7.0f),
                floatArrayOf(2.0f, 5.0f, 8.0f),
                floatArrayOf(3.0f, 6.0f, 9.0f))
        assertContentEquals(expected, A.T.toArray())
        assertContentEquals(A.toArray(), A.T.T.toArray())
        A.transpose()
        assertContentEquals(expected, A.toArray())
    }

    @Test
    fun getRow() {
        val A = matrix(floatArrayOf(2.0f, 3.0f),
                floatArrayOf(4.0f, 6.0f))
        assertContentEquals(floatArrayOf(4.0f, 6.0f), A[1].toFloatArray())
    }

    @Test
    fun setRow() {
        val A = matrix(floatArrayOf(1.0f, 2.0f, 3.0f),
                floatArrayOf(4.0f, 5.0f, 6.0f),
                floatArrayOf(7.0f, 8.0f, 9.0f))

        A[1] = vector(-1f, 2f, 0f)
        val expected = arrayOf(
                floatArrayOf(1.0f, 2.0f, 3.0f),
                floatArrayOf(-1.0f, 2.0f, 0.0f),
                floatArrayOf(7.0f, 8.0f, 9.0f))
        assertContentEquals(expected, A.toArray())
    }

    @Test
    fun squareMatrixTimesVector() {
        val v = vector(-1.0f, 2.0f, 0.0f)
        val A = matrix(
                floatArrayOf(1.0f, 2.0f, 3.0f),
                floatArrayOf(4.0f, 5.0f, 6.0f),
                floatArrayOf(7.0f, 8.0f, 9.0f))
        val result = A * v
        assertContentEquals(floatArrayOf(3.0f, 6.0f, 9.0f), result.toFloatArray())
    }

    @Test
    fun matrixTimesVector() {
        val v = vector(-1.0f, 2.0f)
        val A = matrix(
                floatArrayOf(1.0f, 4.0f),
                floatArrayOf(2.0f, 5.0f),
                floatArrayOf(3.0f, 6.0f))
        val result = A * v
        assertContentEquals(floatArrayOf(7.0f, 8.0f, 9.0f), result.toFloatArray())
    }
}
