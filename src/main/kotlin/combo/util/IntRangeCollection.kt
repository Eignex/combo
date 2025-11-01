package combo.util

import combo.math.permutation
import kotlinx.serialization.Serializable
import kotlin.random.Random

/**
 * A collection of all items between [min] and [max] (inclusive).
 */
@Serializable
data class IntRangeCollection(val min: Int, val max: Int) : IntList {

    init {
        assert(min <= max)
    }

    override var size: Int = kotlin.math.max(0, 1 + (max - min))
        private set

    override fun contains(value: Int): Boolean = value in min..max

    override fun indexOf(value: Int): Int {
        return if (value in min..max) value - min
        else -1
    }

    override fun get(index: Int): Int {
        if (index in 0..(max - min)) return min + index
        else throw IndexOutOfBoundsException("$index")
    }

    override fun toArray(): IntArray {
        val array = IntArray(size)
        var k = 0
        for (i in this)
            array[k++] = i
        return array
    }

    override fun map(transform: (Int) -> Int): IntRangeCollection {
        val min1 = transform.invoke(min)
        val max1 = transform.invoke(max)
        return if (min1 < max1) IntRangeCollection(min1, max1)
        else IntRangeCollection(max1, min1)
    }

    override fun iterator() = (min..max).iterator()

    override fun permutation(rng: Random): IntIterator {
        var i = 0
        val perm = permutation(size, rng)
        return object : IntIterator() {
            override fun hasNext() = i < size
            override fun nextInt(): Int {
                if (i >= size) throw NoSuchElementException()
                return min + perm.encode(i++)
            }
        }
    }

    override fun random(rng: Random): Int {
        if (isEmpty()) throw NoSuchElementException()
        return min + rng.nextInt(size)
    }

    override fun copy() = IntRangeCollection(min, max)

    override fun toString() = "[$min:$max]"
}
