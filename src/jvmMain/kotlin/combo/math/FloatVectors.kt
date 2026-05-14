package combo.math

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

/**
 * Sparse vector backed by parallel index/value arrays. [indices] holds the feature
 * positions that are nonzero; [values] holds those nonzero values in matching order.
 * Linear lookup — fine for the bandit's typical use case (sparse feature vectors from
 * nominal-heavy spaces, where each sample touches a handful of indicator slots out of
 * hundreds-to-thousands).
 *
 * Vectors are constructed-once: [set] only updates existing nonzero positions, never
 * introduces new ones, so [indices] is immutable after construction. Callers that need
 * to add nonzero slots should materialise a fresh sparse vector.
 */
class FloatSparseVector(override val size: Int, val indices: IntArray, val values: FloatArray) : Vector {

    init {
        require(indices.size == values.size) {
            "indices/values must align: ${indices.size} vs ${values.size}"
        }
    }

    override val sparse: Boolean get() = true

    override infix fun dot(v: VectorView): Float {
        var sum = 0f
        for (k in indices.indices) sum += v[indices[k]] * values[k]
        return sum
    }

    override fun norm2(): Float {
        var sum = 0f
        for (v in values) sum += v * v
        return sqrt(sum)
    }

    override fun sum() = values.sum()

    override fun get(i: Int): Float {
        val pos = positionOf(i)
        return if (pos < 0) 0f else values[pos]
    }

    override fun set(i: Int, x: Float) {
        val pos = positionOf(i)
        if (pos < 0) throw UnsupportedOperationException(
            "FloatSparseVector position $i is not in the index — sparse vectors are constructed-once",
        )
        values[pos] = x
    }

    private fun positionOf(i: Int): Int {
        for (k in indices.indices) if (indices[k] == i) return k
        return -1
    }

    override fun iterator(): IntIterator = indices.iterator()

    override fun toFloatArray() = FloatArray(size).also {
        for (k in indices.indices) it[indices[k]] = values[k]
    }

    override fun copy() = FloatSparseVector(size, indices.copyOf(), values.copyOf())
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
    ) = FloatSparseVector(size, indices, values)
}
