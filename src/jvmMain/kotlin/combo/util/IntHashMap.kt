package combo.util

import com.eignex.kpermute.intPermutation
import kotlin.random.Random

/**
 * Specialized open addressing hash table for storing ints. It improves over default Java implementation a lot.
 * It uses linear probing in order to support backshifting remove operation (so that no tombstone marker is needed).
 * Finally, inserts use the Robin Hood Hash method to stabilize performance.
 * This is also an [IntCollection] over the keys of the table.
 */
class IntHashMap private constructor(private var table: LongArray, size: Int, val nullKey: Int = 0) : IntCollection {

    constructor(initialSize: Int = 4, nullKey: Int = 0) : this(
        LongArray(IntCollection.tableSizeFor(initialSize)) {
            entry(nullKey, 0)
        },
        0, nullKey
    )

    override var size: Int = size
        private set

    private var threshold = IntCollection.nextThreshold(table.size)

    private var mask = table.size - 1

    fun isEmpty() = size == 0

    override fun copy() = IntHashMap(table.copyOf(), size, nullKey)

    fun clear() {
        table = LongArray(IntCollection.tableSizeFor(4))
        if (nullKey != 0) {
            val v = entry(nullKey, 0)
            table.forEachIndexed { i, _ -> table[i] = v }
        }
        size = 0
        threshold = IntCollection.nextThreshold(table.size)
        mask = table.size - 1
    }

    override fun contains(value: Int) = table[linearProbe(value)].key() != nullKey

    fun values(): IntArray {
        val result = IntArray(size)
        var ix = 0
        for (i in table.indices)
            if (table[i].key() != nullKey) result[ix++] = table[i].value()
        return result
    }

    /**
     * Map keys in the hashmap.
     */
    override fun map(transform: (Int) -> Int): IntHashMap {
        val mapped = IntHashMap(size, nullKey)
        for (i in table) {
            if (i.key() != nullKey) mapped.add(entry(transform(i.key()), i.value()))
        }
        return mapped
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
                while (table[perm.encode(ptr)].key() == nullKey)
                    ptr = (ptr + 1) and mask
                return table[perm.encode(ptr)].key().also {
                    ptr = (ptr + 1) and mask
                }
            }
        }
    }

    override fun random(rng: Random): Int {
        if (isEmpty()) throw NoSuchElementException()
        while (true) {
            val k = rng.nextInt(table.size)
            if (table[k].key() != nullKey) return table[k].key()
        }
    }

    /**
     * Iterate over the keys in the map.
     */
    override fun iterator(): IntIterator {
        return object : IntIterator() {
            private var seen = 0
            private var ptr = 0
            override fun hasNext() = seen < size

            override fun nextInt(): Int {
                while (table[ptr].key() == nullKey)
                    ptr++
                seen++
                return table[ptr++].key()
            }
        }
    }

    /**
     * Iterate over the key/value entries in the map. Use [value] and [key] to extract the key/value from the longs.
     */
    fun entryIterator(): LongIterator {
        return object : LongIterator() {
            private var seen = 0
            private var ptr = 0
            override fun hasNext() = seen < size

            override fun nextLong(): IntEntry {
                while (table[ptr].key() == nullKey)
                    ptr++
                seen++
                return table[ptr++]
            }
        }
    }

    fun add(entry: IntEntry) = set(entry.key(), entry.value())

    operator fun get(key: Int, default: Int = 0): Int {
        assert(key != nullKey)
        val entry = table[linearProbe(key)]
        return if (entry.key() != nullKey) {
            entry.value()
        } else {
            default
        }
    }

    operator fun set(key: Int, value: Int): Int {
        assert(key != nullKey)

        val p = linearProbe(key)
        val oldEntry = table[p]

        if (oldEntry.key() == nullKey) {
            // Resize if needed
            if (size + 1 >= threshold) {
                val old = table
                table = LongArray(IntCollection.tableSizeFor(size + 1) * 2)
                threshold = IntCollection.nextThreshold(table.size)
                mask = table.size - 1
                if (nullKey != 0) {
                    table.forEachIndexed { i, _ -> table[i] = entry(nullKey, 0) }
                }
                size = 0
                for (i in old.indices)
                    if (old[i].key() != nullKey) add(old[i])
            }
            size++

            // Perform the Robin Hood hash insertion step
            var e = entry(key, value)
            var ix = IntCollection.spread(key) and mask
            var dist = 0
            while (true) {
                if (table[ix].key() == nullKey) {
                    table[ix] = e
                    return 0
                }

                val currentDist = probeDistance(ix)
                if (currentDist < dist) {
                    dist = currentDist
                    val tmp = table[ix]
                    table[ix] = e
                    e = tmp
                }

                ix = (ix + 1) and mask
                dist++
            }
        } else {
            table[p] = entry(key, value)
            return oldEntry.value()
        }
    }

    fun remove(ix: Int, default: Int = 0): Int {
        var p = linearProbe(ix)
        if (table[p].key() == nullKey) return default
        size--

        var j = p
        val oldValue = table[p].value()
        table[p] = entry(nullKey, 0)
        while (true) {
            j = (j + 1) and mask
            if (table[j].key() == nullKey) break
            val k = IntCollection.spread(table[j].key()) and mask
            if ((j > p && (k <= p || k > j)) || (j < p && (k <= p && k > j))) {
                table[p] = table[j]
                p = j
            }
            table[p] = entry(nullKey, 0)
        }
        return oldValue
    }

    private fun probeDistance(pos: Int): Int {
        val desired = IntCollection.spread(table[pos].key()) and mask
        val dist = pos - desired
        return if (dist < 0) {
            table.size + dist
        } else {
            dist
        }
    }

    private fun linearProbe(key: Int): Int {
        var j = IntCollection.spread(key) and mask
        while ((table[j].key() != nullKey && table[j].key() != key))
            j = (j + 1) and mask
        return j
    }

    override fun toString() = entryIterator().asSequence().joinToString(", ", "{", "}") { "${it.key()}=${it.value()}" }
}
