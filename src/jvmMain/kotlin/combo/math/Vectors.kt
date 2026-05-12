@file:JvmName("Vectors")

package combo.math

import combo.util.transformArray
import kotlin.jvm.JvmName
import kotlin.math.roundToInt
import kotlin.math.sqrt

interface VectorView : Iterable<Int> {
    val size: Int
    val sparse: Boolean

    operator fun get(i: Int): Float
    infix fun dot(v: VectorView) = fold(0f) { sum, i -> sum + this[i] * v[i] }
    fun norm2(): Float = sqrt(sumBy { it * it })
    fun sum(): Float = sumBy { it }

    /**
     * Iterates over the non-zero elements index.
     */
    override fun iterator(): IntIterator = (0 until size).iterator()

    operator fun plus(v: VectorView) = v.vectorCopy().also { it.transformIndexed { i, f -> this[i] + f } }
    operator fun minus(v: VectorView) = v.vectorCopy().also { it.transformIndexed { i, f -> this[i] - f } }
    operator fun times(v: VectorView) = v.vectorCopy().also { it.transformIndexed { i, f -> this[i] * f } }
    operator fun div(v: VectorView) = v.vectorCopy().also { it.transformIndexed { i, f -> this[i] / f } }

    operator fun plus(f: Float) = vectorCopy().apply { transform { it + f } }
    operator fun minus(f: Float) = vectorCopy().apply { transform { it - f } }
    operator fun times(f: Float) = vectorCopy().apply { transform { it * f } }
    operator fun div(f: Float) = vectorCopy().apply { transform { it / f } }

    fun copy(): VectorView
    fun vectorCopy(): Vector
    fun asVector(): Vector = vectorCopy()

    /**
     * Returns a copy of the backing data.
     */
    fun toFloatArray(): FloatArray

    fun toIntArray() = IntArray(size) { get(it).roundToInt() }
}

interface Vector : VectorView {
    operator fun set(i: Int, x: Float)
    override fun copy(): Vector

    fun add(f: Float) = transform { it + f }
    fun subtract(f: Float) = transform { it - f }
    fun multiply(f: Float) = transform { it * f }
    fun divide(f: Float) = transform { it / f }

    fun add(v: VectorView) = transformIndexed { i, d -> d + v[i] }
    fun subtract(v: VectorView) = transformIndexed { i, d -> d - v[i] }
    fun multiply(v: VectorView) = transformIndexed { i, d -> d * v[i] }
    fun divide(v: VectorView) = transformIndexed { i, d -> d / v[i] }

    fun assign(v: VectorView) {
        for (i in 0 until v.size) this[i] = v[i]
    }
}

inline val VectorView.indices get() = 0 until size

interface Matrix {
    val rows: Int
    val cols: Int

    operator fun set(i: Int, j: Int, x: Float)
    operator fun get(i: Int, j: Int): Float
    operator fun get(row: Int): VectorView
    operator fun set(row: Int, values: VectorView)

    fun copy(): Matrix

    operator fun times(v: VectorView): Vector

    /**
     * Inline transpose.
     */
    fun transpose()

    val T: Matrix
        get() = copy().apply { transpose() }

    /**
     * Returns a copy of the backing data.
     */
    fun toArray(): Array<FloatArray>
}

interface VectorFactory {
    fun zeroMatrix(rows: Int, columns: Int = rows): Matrix
    fun zeroVector(size: Int): Vector
    fun matrix(values: Array<FloatArray>): Matrix
    fun vector(values: FloatArray): Vector
    fun sparseVector(size: Int, values: FloatArray, indices: IntArray): Vector = zeroVector(size).apply {
        for (i in indices) this[i] = values[i]
    }
}

var vectors: VectorFactory = FloatVectorFactory

val EMPTY_VECTOR = FloatVector(0)
val EMPTY_MATRIX = FloatMatrix(0)

fun VectorView.toIntArray(delta: Float, gcd: Boolean): IntArray {
    val array = IntArray(size) {
        (this[it] / delta).roundToInt()
    }
    if (gcd) {
        val g = gcdAll(*array)
        if (g > 1) array.transformArray { it / g }
    }
    return array
}

inline fun Vector.transform(transform: (Float) -> Float) {
    for (i in indices)
        this[i] = transform(this[i])
}

inline fun Vector.transformIndexed(transform: (Int, Float) -> Float) {
    for (i in indices)
        this[i] = transform(i, this[i])
}

inline fun VectorView.sumBy(selector: (Float) -> Float): Float {
    var sum = 0.0f
    for (i in this) {
        sum += selector(this[i])
    }
    return sum
}
