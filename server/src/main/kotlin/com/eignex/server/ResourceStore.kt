package com.eignex.server

import com.eignex.klause.util.Cancellation
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.job
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

internal class OperationControl(private val job: Job) {
    val cancellation = Cancellation { !job.isActive }
    fun checkpoint() = job.ensureActive()
}

internal class OperationRunner(workers: Int = 2, queueCapacity: Int = 64) : AutoCloseable {
    private val admission = Semaphore(workers + queueCapacity)
    private val threadNumber = AtomicInteger()
    private val executor = Executors.newFixedThreadPool(workers) { task ->
        Thread(task, "eignex-operation-${threadNumber.incrementAndGet()}").apply { isDaemon = true }
    }
    private val dispatcher = executor.asCoroutineDispatcher()
    private val lock = Any()
    private val active = mutableSetOf<Job>()
    private var closed = false

    suspend fun <T> run(timeoutMillis: Long = 30_000, action: suspend (OperationControl) -> T): T {
        validateTimeout(timeoutMillis)
        if (!admission.tryAcquire()) capacityExceeded()
        try {
            return withTimeout(timeoutMillis) {
                val job = currentCoroutineContext().job
                synchronized(lock) {
                    if (closed) capacityExceeded()
                    active += job
                }
                try {
                    withContext(dispatcher) {
                        val control = OperationControl(job)
                        control.checkpoint()
                        action(control)
                    }
                } finally {
                    synchronized(lock) { active.remove(job) }
                }
            }
        } finally {
            admission.release()
        }
    }

    override fun close() {
        synchronized(lock) {
            closed = true
            active.toList().forEach { it.cancel() }
            dispatcher.close()
        }
    }
}

internal interface ManagedResource : AutoCloseable {
    val type: String
    override fun close() = Unit
}

internal class ResourceStore<T : ManagedResource>(
    private val runner: OperationRunner,
    private val capacity: Int = 1000,
) : AutoCloseable {
    private class Entry<T : ManagedResource>(val id: String, val resource: T) {
        val mutex = Mutex()
        @Volatile var retired = false
        private var disposed = false

        fun dispose() {
            if (!disposed) {
                disposed = true
                resource.close()
            }
        }
    }

    private val lock = Any()
    private val entries = linkedMapOf<String, Entry<T>>()
    private var closed = false

    suspend fun create(timeoutMillis: Long = 30_000, factory: (OperationControl) -> T): ResourceInfo =
        runner.run(timeoutMillis) { control ->
            synchronized(lock) { if (closed || entries.size >= capacity) capacityExceeded() }
            val resource = factory(control)
            var registered = false
            try {
                control.checkpoint()
                synchronized(lock) {
                    if (closed || entries.size >= capacity) capacityExceeded()
                    val id = UUID.randomUUID().toString()
                    entries[id] = Entry(id, resource)
                    registered = true
                    ResourceInfo(id, resource.type)
                }
            } finally {
                if (!registered) resource.close()
            }
        }

    fun list(): List<ResourceInfo> = synchronized(lock) {
        entries.values.map { ResourceInfo(it.id, it.resource.type) }
    }

    suspend fun <R> access(id: String, timeoutMillis: Long = 30_000, action: (T, OperationControl) -> R): R {
        val entry = synchronized(lock) { entries[id] ?: notFound() }
        return runner.run(timeoutMillis) { control ->
            entry.mutex.withLock {
                try {
                    if (entry.retired) notFound()
                    control.checkpoint()
                    action(entry.resource, control)
                } finally {
                    if (entry.retired) entry.dispose()
                }
            }
        }
    }

    suspend fun delete(id: String) {
        val entry = synchronized(lock) { entries[id] ?: notFound() }
        runner.run { _ ->
            entry.mutex.withLock {
                try {
                    if (entry.retired) notFound()
                    synchronized(lock) {
                        entry.retired = true
                        entries.remove(id)
                    }
                } finally {
                    if (entry.retired) entry.dispose()
                }
            }
        }
    }

    override fun close() {
        val retired = synchronized(lock) {
            closed = true
            entries.values.toList().also {
                it.forEach { entry -> entry.retired = true }
                entries.clear()
            }
        }
        for (entry in retired) {
            if (entry.mutex.tryLock()) {
                try { entry.dispose() } finally { entry.mutex.unlock() }
            }
        }
    }
}
