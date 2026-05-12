@file:JvmName("IntCollections")

package combo.util

import kotlin.jvm.JvmName
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random

interface IntCollection : Iterable<Int> {
    val size: Int
    fun copy(): IntCollection
    operator fun contains(value: Int): Boolean

    fun toArray(): IntArray {
        val arr = IntArray(size)
        for ((i, v) in this.withIndex()) arr[i] = v
        return arr
    }

    fun map(transform: (Int) -> Int): IntCollection

    override fun iterator(): IntIterator
    fun permutation(rng: Random): IntIterator

    fun random(rng: Random): Int

    @Suppress("NOTHING_TO_INLINE")
    companion object {

        inline fun spread(i: Int) = i * 0x7F4A7C15

        const val LOAD_FACTOR = 0.85f

        fun tableSizeFor(size: Int): Int {
            return max(2, Int.power2((size.toFloat() / LOAD_FACTOR).toInt()))
        }

        fun nextThreshold(currentSize: Int): Int {
            return max(2, (currentSize * LOAD_FACTOR).toInt())
        }
    }
}

interface IntList : IntCollection {
    operator fun get(index: Int): Int
    fun indexOf(value: Int): Int
}

interface MutableIntCollection : IntCollection {
    override fun copy(): MutableIntCollection
    fun clear()
    override fun map(transform: (Int) -> Int): MutableIntCollection
    fun add(value: Int): Boolean
    fun addAll(values: IntArray) = values.fold(false) { any, it -> this.add(it) || any }
    fun addAll(values: Iterable<Int>) = values.fold(false) { any, it -> this.add(it) || any }
    fun remove(value: Int): Boolean
    fun removeAll(values: Iterable<Int>) = values.fold(false) { any, it -> this.remove(it) || any }
}

operator fun MutableIntCollection.plus(collection: IntCollection): IntCollection {
    return copy().apply { addAll(collection) }
}

operator fun MutableIntCollection.plus(value: Int): MutableIntCollection {
    return copy().apply { add(value) }
}

/**
 * Immutable list collection.
 */
fun intListOf(vararg array: Int): IntList {
    if (array.size > 1) {
        var min = Int.MAX_VALUE
        var max = Int.MIN_VALUE
        var doRange = true
        for ((i, v) in array.withIndex()) {
            min = min(v, min)
            max = max(v, max)
            if (max - min != i) {
                doRange = false
                break
            }
        }
        if (doRange) return IntRangeCollection(min, max)
    }
    val list = IntArrayList()
    list.addAll(array)
    return list
}

/**
 * Immutable collection. This may not contain 0.
 */
fun collectionOf(vararg array: Int): IntCollection {
    // Check for IntRange
    var min = Int.MAX_VALUE
    var max = Int.MIN_VALUE
    var doRange = true
    for ((i, v) in array.withIndex()) {
        assert(v != 0)
        min = min(v, min)
        max = max(v, max)
        if (max - min != i) {
            doRange = false
            break
        }
    }
    return when {
        array.isEmpty() -> EmptyCollection
        array.size == 1 -> SingletonIntCollection(array[0])
        doRange -> IntRangeCollection(min, max)
        array.size > 20 -> IntHashSet().apply { addAll(array) }
        else -> IntArrayList(array)
    }
}

fun unionCollection(a: IntCollection, value: Int): IntCollection {
    return if (a is IntRangeCollection && (value + 1 in a || value - 1 in a)) {
        IntRangeCollection(min(a.min, value), max(a.max, value))
    } else {
        IntUnionCollection(a, IntArrayList(intArrayOf(value)))
    }
}

fun IntCollection.mutableCopy(nullValue: Int): MutableIntCollection =
    when {
        this is MutableIntCollection -> this.copy()
        else -> (if (this.size > 20) IntHashSet(nullValue = nullValue) else IntArrayList()).apply {
            addAll(this@mutableCopy)
        }
    }

fun IntCollection.isEmpty() = size == 0
fun IntCollection.isNotEmpty() = size > 0

object EmptyCollection : IntCollection {
    override val size = 0

    override fun copy() = IntHashSet()
    override operator fun contains(value: Int) = false
    override fun toArray() = EMPTY_INT_ARRAY
    override fun map(transform: (Int) -> Int) = this
    override fun iterator() = object : IntIterator() {
        override fun hasNext() = false
        override fun nextInt() = throw NoSuchElementException()
    }

    override fun permutation(rng: Random) = iterator()
    override fun random(rng: Random) = throw NoSuchElementException()
}

typealias IntEntry = Long

fun IntEntry.key() = this.toInt()
fun IntEntry.value() = (this ushr (Int.SIZE_BITS)).toInt()
fun entry(key: Int, value: Int) = (value.toLong() shl Int.SIZE_BITS) or (key.toLong() and 0xFFFFFFFFL)

class SingletonIntCollection(val value: Int) : IntCollection {
    override val size: Int get() = 1
    override fun copy() = SingletonIntCollection(value)
    override fun contains(value: Int) = this.value == value
    override fun map(transform: (Int) -> Int) = SingletonIntCollection(transform.invoke(value))
    override fun iterator() = object : IntIterator() {
        var consumed = false
        override fun hasNext() = !consumed
        override fun nextInt(): Int {
            if (consumed) throw NoSuchElementException()
            consumed = true
            return value
        }
    }

    override fun permutation(rng: Random) = iterator()
    override fun random(rng: Random) = value
    override fun toString() = "[$value]"
}
