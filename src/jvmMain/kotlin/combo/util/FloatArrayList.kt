package combo.util

class FloatArrayList private constructor(private var array: FloatArray, size: Int) : Iterable<Float> {

    constructor(initialSize: Int = 4) : this(FloatArray(initialSize), 0)
    constructor(array: FloatArray) : this(array, array.size)

    var size: Int = size
        private set

    fun copy() = FloatArrayList(array.copyOf(), size)

    fun clear() {
        size = 0
    }

    operator fun contains(value: Float) = indexOf(value) >= 0

    operator fun get(index: Int) = array[index]

    fun indexOf(value: Float): Int {
        for (i in 0 until size)
            if (array[i] == value) return i
        return -1
    }

    fun toArray() = array.copyOfRange(0, size)

    fun map(transform: (Float) -> Float) =
        FloatArrayList(
            FloatArray(size) { i ->
                transform(this.array[i])
            }
        )

    override fun iterator() = object : FloatIterator() {
        private var ptr = 0
        override fun hasNext() = ptr < size
        override fun nextFloat(): Float {
            if (ptr >= size) throw NoSuchElementException()
            return array[ptr++]
        }
    }

    fun add(value: Float): Boolean {
        if (array.size == size) {
            array = array.copyOf(kotlin.math.max(array.size, 1) * 2)
        }
        array[size++] = value
        return true
    }

    fun removeAt(ix: Int): Float {
        val v = array[ix]
        size--
        for (i in ix until size)
            array[i] = array[i + 1]
        array[size] = 0f
        return v
    }

    fun remove(value: Float): Boolean {
        val indexOf = indexOf(value)
        if (indexOf >= 0) {
            size--
            for (i in indexOf until size)
                array[i] = array[i + 1]
            array[size] = 0f
            return true
        }
        return false
    }

    override fun toString() = joinToString(", ", "[", "]")
}
