package combo.util

import java.util.concurrent.locks.Lock
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.withLock

@Suppress("UNCHECKED_CAST")
class RandomListCache<E>(val maxSize: Int, randomSeed: Int) {

    private val readWriteLock = ReentrantReadWriteLock()
    private val writeLock: Lock = readWriteLock.writeLock()

    private val randomSequence = RandomSequence(randomSeed)

    @PublishedApi
    internal val readLock: Lock = readWriteLock.readLock()

    @PublishedApi
    internal val array = arrayOfNulls<Any>(maxSize)

    @PublishedApi
    internal var size: Int = 0

    fun isFull() = readLock.withLock { size == maxSize }

    fun put(e: E) {
        val rng = randomSequence.next()
        writeLock.withLock {
            if (size < maxSize) {
                array[size++] = e
            } else {
                array[rng.nextInt(maxSize)] = e
            }
        }
    }

    inline fun forEach(action: (E) -> Unit) =
        readLock.withLock {
            for (i in 0 until size) {
                val e = array[i] as E
                action.invoke(e)
            }
        }

    inline fun find(predicate: (E) -> Boolean): E? =
        readLock.withLock {
            for (i in 0 until size) {
                val e = array[i] as E
                if (predicate.invoke(e)) {
                    return e
                }
            }
            return null
        }
}

@Suppress("UNCHECKED_CAST")
class RandomMapCache<K, V>(val maxSize: Int, randomSeed: Int = System.currentTimeMillis().toInt()) {

    private val readWriteLock = ReentrantReadWriteLock()
    private val writeLock: Lock = readWriteLock.writeLock()

    var hits: Int = 0
    var requests: Int = 0

    private val randomSequence = RandomSequence(randomSeed)
    private val map = HashMap<K, Int>()
    private val readLock: Lock = readWriteLock.readLock()
    private val keys = arrayOfNulls<Any>(maxSize)
    private val values = arrayOfNulls<Any>(maxSize)

    fun put(key: K, value: V) = writeLock.withLock {
        val oldIx = map[key]
        val rng = randomSequence.next()
        when {
            oldIx != null -> values[oldIx] = value
            map.size < maxSize -> {
                val ix = map.size
                keys[ix] = key
                values[ix] = value
                map[key] = ix
            }
            else -> {
                val evicted = rng.nextInt(maxSize)
                map.remove(keys[evicted] as K)
                map[key] = evicted
                keys[evicted] = key
                values[evicted] = value
            }
        }
    }

    fun get(key: K): V? = readLock.withLock {
        return values[map[key] ?: return null] as V?
    }

    fun getOrPut(key: K, create: () -> V): V {
        requests++
        val get = get(key)
        if (get != null) hits++
        return get ?: create().also { put(key, it) }
    }
}
