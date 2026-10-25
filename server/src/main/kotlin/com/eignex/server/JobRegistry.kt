package com.eignex.server

import com.eignex.klause.util.Cancellation
import kotlinx.serialization.json.JsonObject
import org.slf4j.LoggerFactory
import java.time.Instant
import java.util.UUID
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.FutureTask
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

internal class JobControl(timeoutMillis: Long) {
    private val deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMillis)
    val cancelled = AtomicBoolean()
    val cancellation = Cancellation { cancelled.get() || expired() || Thread.currentThread().isInterrupted }

    private fun expired(): Boolean = System.nanoTime() - deadline >= 0

    fun checkpoint() {
        if (cancelled.get() || Thread.currentThread().isInterrupted) throw JobCancelled()
        if (expired()) throw JobTimedOut()
    }
}

internal class JobCancelled : RuntimeException()
internal class JobTimedOut : RuntimeException()
internal class JobCapacityExceeded : RuntimeException()
internal class JobConflict(message: String) : RuntimeException(message)

internal class JobRegistry(
    workers: Int = 2,
    queueCapacity: Int = 64,
    private val retainedCapacity: Int = 1000,
    private val execute: (JobRequest, JobControl) -> JsonObject = ::executeJob,
) : AutoCloseable {
    private class Entry(var submission: Submission?, var snapshot: JobSnapshot) {
        var control: JobControl? = null
        lateinit var task: FutureTask<Unit>
    }

    private val log = LoggerFactory.getLogger(JobRegistry::class.java)
    private val lock = Any()
    private val entries = linkedMapOf<String, Entry>()
    private var closed = false
    private val threadNumber = AtomicInteger()
    private val executor = ThreadPoolExecutor(
        workers, workers, 0L, TimeUnit.MILLISECONDS, ArrayBlockingQueue(queueCapacity),
        { task -> Thread(task, "eignex-job-${threadNumber.incrementAndGet()}").apply { isDaemon = true } },
        ThreadPoolExecutor.AbortPolicy(),
    )

    fun submit(submission: Submission): JobSnapshot = synchronized(lock) {
        if (closed || entries.size >= retainedCapacity) throw JobCapacityExceeded()
        val snapshot = JobSnapshot(
            UUID.randomUUID().toString(), submission.request.operation, JobStatus.Queued, Instant.now().toString(),
        )
        val entry = Entry(submission, snapshot)
        entry.task = FutureTask { run(entry) }
        entries[snapshot.id] = entry
        try {
            executor.execute(entry.task)
        } catch (_: RejectedExecutionException) {
            entries.remove(snapshot.id)
            throw JobCapacityExceeded()
        }
        snapshot
    }

    fun list(): List<JobSnapshot> = synchronized(lock) { entries.values.map { it.snapshot } }
    fun get(id: String): JobSnapshot? = synchronized(lock) { entries[id]?.snapshot }

    fun cancel(id: String): JobSnapshot? = synchronized(lock) {
        val entry = entries[id] ?: return null
        if (!entry.snapshot.status.terminal) {
            entry.control?.cancelled?.set(true)
            entry.snapshot = entry.snapshot.copy(status = JobStatus.Cancelled, finishedAt = Instant.now().toString())
            entry.task.cancel(true)
            executor.remove(entry.task)
            entry.submission = null
        }
        entry.snapshot
    }

    fun delete(id: String): Boolean = synchronized(lock) {
        val entry = entries[id] ?: return false
        if (!entry.snapshot.status.terminal) throw JobConflict("cancel the job before deleting it")
        entries.remove(id)
        true
    }

    private fun run(entry: Entry) {
        val (submission, control) = synchronized(lock) {
            if (entry.snapshot.status != JobStatus.Queued) return
            val submission = checkNotNull(entry.submission)
            val control = JobControl(submission.timeoutMillis).also {
                entry.control = it
                entry.snapshot = entry.snapshot.copy(status = JobStatus.Running, startedAt = Instant.now().toString())
            }
            entry.submission = null
            submission to control
        }
        try {
            control.checkpoint()
            val result = execute(submission.request, control)
            control.checkpoint()
            finish(entry, JobStatus.Succeeded, result)
        } catch (_: JobCancelled) {
            finish(entry, JobStatus.Cancelled)
        } catch (_: JobTimedOut) {
            finish(entry, JobStatus.TimedOut)
        } catch (error: Throwable) {
            when {
                control.cancelled.get() -> finish(entry, JobStatus.Cancelled)
                error is IllegalArgumentException -> finish(
                    entry, JobStatus.Failed, error = ApiError("invalid_job", error.message ?: "invalid job input"),
                )
                else -> {
                    log.error("Job {} failed", entry.snapshot.id, error)
                    finish(entry, JobStatus.Failed, error = ApiError("execution_failed", "job execution failed"))
                }
            }
        }
    }

    private fun finish(entry: Entry, status: JobStatus, result: JsonObject? = null, error: ApiError? = null) {
        synchronized(lock) {
            if (!entry.snapshot.status.terminal) {
                entry.snapshot = entry.snapshot.copy(
                    status = status, finishedAt = Instant.now().toString(), result = result, error = error,
                )
            }
            entry.control = null
        }
    }

    override fun close() {
        synchronized(lock) {
            closed = true
            entries.keys.toList().forEach(::cancel)
            executor.shutdownNow()
        }
    }
}
