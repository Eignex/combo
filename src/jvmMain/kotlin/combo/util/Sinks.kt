package combo.util

import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

interface Sink<T> {

    /**
     * Offer the element to the sink, returning true if the element is accepted or false if rejected.
     */
    fun offer(element: T): Boolean

    /**
     * Adds element to the sink, waiting if necessary until the sink accepts.
     */
    fun add(element: T) {
        offer(element)
    }

    /**
     * Returns an element from the sink if there is one, otherwise null. Will not block.
     */
    fun remove(): T?

    /**
     * Clears the buffer and returns a read-only view of the result.
     */
    fun drain(min: Int = -1): Iterable<T>
}

class BlockingSink<T>(arraySize: Int) : Sink<T> {

    private val lock = ReentrantLock()

    @Suppress("UNCHECKED_CAST")
    private val array = arrayOfNulls<Any?>(arraySize) as Array<T?>
    private var size: Int = 0
    private val notFull = lock.newCondition()
    private val notEmpty = lock.newCondition()

    private fun enqueue(element: T) {
        array[size++] = element
        notEmpty.signal()
    }

    override fun add(element: T) {
        lock.withLock {
            while (size == array.size)
                notFull.await()
            enqueue(element)
        }
    }

    override fun offer(element: T): Boolean {
        lock.withLock {
            if (size == array.size) {
                return false
            }
            enqueue(element)
            return true
        }
    }

    override fun drain(min: Int): Iterable<T> {
        lock.withLock {
            while (size < min)
                notEmpty.await()
            val list = ArrayList<T>(size)
            var i = 0
            while (i < size) {
                list.add(array[i]!!)
                array[i] = null
                i++
            }
            size = 0
            notFull.signalAll()
            return list
        }
    }

    override fun remove(): T? {
        lock.withLock {
            if (size > 0) {
                val i = size - 1
                val t = array[i]
                array[i] = null
                size--
                notFull.signal()
                return t
            }
            return if (size > 0) array[size - 1] else null
        }
    }
}

class LockingSink<T> : Sink<T> {

    private var tail: Node<T>? = null
    private val lock = ReentrantLock()
    private val notEmpty = lock.newCondition()
    private var size = 0

    override fun offer(element: T): Boolean {
        lock.withLock {
            tail = Node(element, tail)
            size++
            notEmpty.signal()
        }
        return true
    }

    override fun drain(min: Int): Iterable<T> {
        lock.withLock {
            while (size < min)
                notEmpty.await()
            size = 0
            val t = tail
            tail = null
            return t ?: emptyList()
        }
    }

    override fun remove(): T? {
        lock.withLock {
            return if (tail != null) {
                val t = tail!!.obj
                tail = tail!!.prev
                size--
                t
            } else {
                null
            }
        }
    }

    private class Node<T>(val obj: T, val prev: Node<T>?) : Iterable<T> {
        override fun iterator(): Iterator<T> = NodeIterator(this)
    }

    private class NodeIterator<T>(var node: Node<T>?) : Iterator<T> {
        override fun hasNext() = node != null
        override fun next(): T {
            if (node == null) throw NoSuchElementException()
            val n = node!!
            val t = n.obj
            node = n.prev
            return t
        }
    }
}

class NonBlockingSink<T> : Sink<T> {

    private val tail: AtomicReference<Node<T>?> = AtomicReference(null)

    override fun offer(element: T): Boolean {
        while (true) {
            val tailNode = tail.get()
            val newTailNode: Node<T> = Node(element, tailNode)
            if (tail.compareAndSet(tailNode, newTailNode)) {
                break
            }
        }
        return true
    }

    override fun drain(min: Int): Iterable<T> {
        return tail.getAndSet(null) ?: emptyList()
    }

    override fun remove(): T? {
        while (true) {
            val tailNode = tail.get()
            if (tail.compareAndSet(tailNode, tailNode?.prev)) {
                return tailNode?.obj
            }
        }
    }
}

private class Node<T>(val obj: T, val prev: Node<T>?) : Iterable<T> {
    override fun iterator(): Iterator<T> = NodeIterator(this)
}

private class NodeIterator<T>(var node: Node<T>?) : Iterator<T> {
    override fun hasNext() = node != null
    override fun next(): T {
        if (node == null) throw NoSuchElementException()
        val n = node!!
        val t = n.obj
        node = n.prev
        return t
    }
}
