package combo.math

import combo.util.IntHashMap
import combo.util.key
import combo.util.sumByFloat
import combo.util.value
import kotlin.math.sqrt

class FloatVector(val array: FloatArray) : Vector {

    constructor(size: Int) : this(FloatArray(size))

    override val size: Int get() = array.size
    override val sparse: Boolean get() = false

    override fun get(i: Int) = array[i]
    override fun set(i: Int, x: Float) {
        array[i] = x
    }

    override infix fun dot(v: VectorView) = array.foldIndexed(0f) { i, dot, d -> dot + d * v[i] }
    override fun sum() = array.sum()

    override fun toFloatArray() = array.copyOf()
    override fun copy() = FloatVector(array.copyOf())
    override fun vectorCopy() = copy()
    override fun asVector() = this
}

class FloatSparseVector(override val size: Int, val values: FloatArray, val index: IntHashMap) : Vector {

    constructor(size: Int, values: FloatArray, indices: IntArray) : this(
        size,
        values,
        IntHashMap(values.size * 2, nullKey = -1).also {
            for (i in values.indices)
                it[indices[i]] = i
        }
    )

    override val sparse: Boolean get() = false

    override infix fun dot(v: VectorView): Float {
        var sum = 0f
        for (l in index.entryIterator()) {
            val i = l.key()
            val j = l.value()
            sum += v[i] * values[j]
        }
        return sum
    }

    override fun norm2() = sqrt(values.sumByFloat { it * it })
    override fun sum() = values.sum()

    override fun get(i: Int): Float {
        val v = index[i, -1]
        return if (v == -1) {
            0f
        } else {
            values[v]
        }
    }

    override fun set(i: Int, x: Float) {
        if (!index.contains(i)) {
            throw UnsupportedOperationException("Can't set new value in sparse vector.")
        } else {
            values[index[i]] = x
        }
    }

    override fun iterator() = index.iterator()

    override fun toFloatArray() = FloatArray(size).also {
        for (l in index.entryIterator()) {
            it[l.key()] = values[l.value()]
        }
    }

    override fun copy() = FloatSparseVector(size, values.copyOf(), index.copy())
    override fun vectorCopy() = copy()
    override fun asVector() = this
}

class FloatMatrix(val array: Array<FloatArray>) : Matrix {

    constructor(size: Int) : this(Array(size) { FloatArray(size) })
    constructor(rows: Int, cols: Int) : this(Array(rows) { FloatArray(cols) })

    override val rows: Int get() = array.size
    override val cols: Int get() = if (array.isEmpty()) 0 else array[0].size

    override operator fun get(i: Int, j: Int) = array[i][j]
    override fun get(row: Int): VectorView = FloatVector(array[row])
    override operator fun set(i: Int, j: Int, x: Float) {
        array[i][j] = x
    }

    override fun set(row: Int, values: VectorView) {
        array[row] = values.toFloatArray()
    }

    override operator fun times(v: VectorView) =
        FloatVector(
            FloatArray(rows) {
                v dot this@FloatMatrix[it]
            }
        )

    override fun transpose() {
        for (i in 0 until rows - 1)
            for (j in (i + 1) until rows) {
                val tmp = this[i, j]
                this[i, j] = this[j, i]
                this[j, i] = tmp
            }
    }

    override fun toArray() = Array(rows) { array[it].copyOf() }
    override fun copy() = FloatMatrix(toArray())
}

object FloatVectorFactory : VectorFactory {
    override fun zeroMatrix(rows: Int, columns: Int) = FloatMatrix(rows, columns)
    override fun zeroVector(size: Int) = FloatVector(size)
    override fun matrix(values: Array<FloatArray>) = FloatMatrix(values)
    override fun vector(values: FloatArray) = FloatVector(values)
    override fun sparseVector(
        size: Int,
        values: FloatArray,
        indices: IntArray
    ) = FloatSparseVector(size, values, indices)
}
