package combo.util

import kotlin.math.min

class ArrayQueue<E> : Iterable<E> {

    @Suppress("UNCHECKED_CAST")
    private var array: Array<E?> = arrayOfNulls<Any?>(4) as Array<E?>
    private var write = 0
    private var read = 0

    private var mask = array.size - 1

    fun add(e: E) {
        array[write] = e
        write = (write + 1) and mask
        if (write == read) {
            write = array.size
            val breakPoint = array.size - read

            @Suppress("UNCHECKED_CAST")
            val a: Array<E?> = arrayOfNulls<Any?>(array.size shl 1) as Array<E?>
            for (i in read until read + breakPoint) a[i - read] = array[i]
            for (i in 0 until read) a[breakPoint + i] = array[i]
            array = a
            read = 0
            mask = array.size - 1
        }
    }

    fun remove() = poll() ?: throw NoSuchElementException()

    fun poll(): E? {
        val e = array[read] ?: return null
        array[read] = null
        read = (read + 1) and mask
        return e
    }

    fun peek(): E? = array[read]

    fun addAll(es: Iterable<E>) {
        for (e in es)
            add(e)
    }

    override fun iterator() = object : Iterator<E> {
        var ptr = read
        override fun hasNext() = array[ptr] != null
        override fun next(): E {
            if (array[ptr] == null) throw NoSuchElementException()
            val value = array[ptr]
            ptr = (ptr + 1) and mask
            return value!!
        }
    }

    val size: Int
        get() = if (write >= read) {
            write - read
        } else {
            array.size - (read - write)
        }
}

class FloatCircleBuffer(private var array: FloatArray) : Iterable<Float> {
    constructor(bufferSize: Int) : this(FloatArray(bufferSize))

    private var write = 0

    fun add(f: Float): Float {
        val old = array[write]
        array[write] = f
        size = min(array.size, size + 1)
        write = (write + 1) % array.size
        return old
    }

    override fun iterator() = object : FloatIterator() {
        private var caret = write % array.size
        private var seen = 0
        override fun hasNext() = seen < size
        override fun nextFloat(): Float {
            if (caret >= size) throw NoSuchElementException()
            val value = array[caret]
            caret = (caret + 1) % array.size
            seen++
            return value
        }
    }

    fun toArray() = array.toTypedArray()

    var size: Int = 0
        private set
}
