package combo.util

import com.eignex.kpermute.intPermutation
import kotlin.random.Random

/**
 * Specialized open addressing hash table for storing ints. It improves over default Java implementation a lot.
 * It uses linear probing in order to support backshifting remove operation (so that no tombstone marker is needed).
 * Finally, inserts use the Robin Hood Hash method to stabilize performance.
 */
class IntHashSet private constructor(
    private var table: IntArray,
    size: Int,
    val nullValue: Int = 0
) : MutableIntCollection {

    /**
     * @param initialSize ensure the capacity of this many items
     * @param nullValue use this to represent null values. This value cannot be added to the set.
     */
    constructor(initialSize: Int = 4, nullValue: Int = 0) : this(
        IntArray(IntCollection.tableSizeFor(initialSize)) { nullValue },
        0,
        nullValue
    )

    override var size: Int = size
        private set

    private var threshold = IntCollection.nextThreshold(table.size)

    private var mask = table.size - 1

    override fun copy() = IntHashSet(table.copyOf(), size, nullValue)

    override fun clear() {
        table = IntArray(IntCollection.tableSizeFor(4))
        if (nullValue != 0) {
            table.forEachIndexed { i, _ -> table[i] = nullValue }
        }
        size = 0
        threshold = IntCollection.nextThreshold(table.size)
        mask = table.size - 1
    }

    override operator fun contains(value: Int) = table[linearProbe(value)] != nullValue

    override fun map(transform: (Int) -> Int): IntHashSet {
        val mapped = IntHashSet(size, nullValue)
        for (i in table) {
            if (i != nullValue) mapped.add(transform(i))
        }
        return mapped
    }

    override fun iterator(): IntIterator {
        return object : IntIterator() {
            private var seen = 0
            private var ptr = 0
            override fun hasNext() = seen < size

            override fun nextInt(): Int {
                if (seen >= size) throw NoSuchElementException()
                seen++
                while (table[ptr] == nullValue)
                    ptr = (ptr + 1) and mask
                return table[ptr].also {
                    ptr = (ptr + 1) and mask
                }
            }
        }
    }

    override fun permutation(rng: Random): IntIterator {
        return object : IntIterator() {
            private var perm = intPermutation(table.size, rng)
            private var seen = 0
            private var ptr = 0
            override fun hasNext() = seen < size

            override fun nextInt(): Int {
                if (seen >= size) throw NoSuchElementException()
                seen++
                while (table[perm.encode(ptr)] == nullValue)
                    ptr = (ptr + 1) and mask
                return table[perm.encode(ptr)].also {
                    ptr = (ptr + 1) and mask
                }
            }
        }
    }

    override fun random(rng: Random): Int {
        if (isEmpty()) throw NoSuchElementException()
        while (true) {
            val k = rng.nextInt(table.size)
            if (table[k] != nullValue) return table[k]
        }
    }

    override fun add(value: Int): Boolean {
        assert(value != nullValue)
        if (table[linearProbe(value)] != nullValue) {
            return false
        }

        // Resize if needed
        if (size + 1 >= threshold) {
            val old = table
            table = IntArray(IntCollection.tableSizeFor(size + 1) * 2)
            threshold = IntCollection.nextThreshold(table.size)
            mask = table.size - 1
            if (nullValue != 0) {
                table.forEachIndexed { i, _ -> table[i] = nullValue }
            }
            size = 0
            for (i in old.indices)
                if (old[i] != nullValue) add(old[i])
        }
        size++

        // Perform the Robin Hood hash insertion step
        var v = value
        var ix = IntCollection.spread(value) and mask
        var dist = 0
        while (true) {
            if (table[ix] == nullValue) {
                table[ix] = v
                return true
            }

            val currentDist = probeDistance(ix)
            if (currentDist < dist) {
                dist = currentDist
                val tmp = table[ix]
                table[ix] = v
                v = tmp
            }

            ix = (ix + 1) and mask
            dist++
        }
    }

    override fun remove(value: Int): Boolean {
        var p = linearProbe(value)
        if (table[p] == nullValue) return false
        size--

        var j = p
        table[p] = nullValue
        while (true) {
            j = (j + 1) and mask
            if (table[j] == nullValue) break
            val k = IntCollection.spread(table[j]) and mask
            if ((j > p && (k <= p || k > j)) || (j < p && (k <= p && k > j))) {
                table[p] = table[j]
                p = j
            }
            table[p] = nullValue
        }
        return true
    }

    private fun probeDistance(pos: Int): Int {
        val desired = IntCollection.spread(table[pos]) and mask
        val dist = pos - desired
        return if (dist < 0) {
            table.size + dist
        } else {
            dist
        }
    }

    private fun linearProbe(value: Int): Int {
        var j = IntCollection.spread(value) and mask
        while ((table[j] != nullValue && table[j] != value))
            j = (j + 1) and mask
        return j
    }

    override fun toString() = joinToString(", ", "[", "]")
}
