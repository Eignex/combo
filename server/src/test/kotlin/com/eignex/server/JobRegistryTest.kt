package com.eignex.server

import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.serialization.json.JsonObject
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class JobRegistryTest {
    private val submission = Submission(KumulantSummary(listOf(1.0)))
    private val empty = JsonObject(emptyMap())

    @Test
    fun `cancelling queued work releases queue capacity`() {
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        JobRegistry(workers = 1, queueCapacity = 1, execute = { _, _ ->
            started.countDown()
            release.await()
            empty
        }).use { jobs ->
            val first = jobs.submit(submission)
            assertTrue(started.await(5, TimeUnit.SECONDS))
            val second = jobs.submit(submission)
            assertEquals(JobStatus.Queued, jobs.get(second.id)?.status)
            assertFailsWith<JobCapacityExceeded> { jobs.submit(submission) }
            assertEquals(JobStatus.Cancelled, jobs.cancel(second.id)?.status)
            val third = jobs.submit(submission)
            release.countDown()
            assertEquals(JobStatus.Succeeded, await(jobs, first.id).status)
            assertEquals(JobStatus.Succeeded, await(jobs, third.id).status)
            assertEquals(JobStatus.Cancelled, jobs.get(second.id)?.status)
        }
    }

    @Test
    fun `late completion cannot overwrite cancellation`() {
        val started = CountDownLatch(1)
        val interrupted = CountDownLatch(1)
        val release = CountDownLatch(1)
        val finished = CountDownLatch(1)
        JobRegistry(workers = 1, execute = { _, _ ->
            started.countDown()
            try {
                release.await()
            } catch (_: InterruptedException) {
                interrupted.countDown()
                release.await()
            }
            finished.countDown()
            empty
        }).use { jobs ->
            val first = jobs.submit(submission)
            assertTrue(started.await(5, TimeUnit.SECONDS))
            assertFailsWith<JobConflict> { jobs.delete(first.id) }
            assertEquals(JobStatus.Cancelled, jobs.cancel(first.id)?.status)
            assertTrue(interrupted.await(5, TimeUnit.SECONDS))
            release.countDown()
            assertTrue(finished.await(5, TimeUnit.SECONDS))
            val next = jobs.submit(submission)
            assertEquals(JobStatus.Succeeded, await(jobs, next.id).status)
            assertEquals(JobStatus.Cancelled, jobs.get(first.id)?.status)
            assertEquals(null, jobs.get(first.id)?.result)
        }
    }

    @Test
    fun `retained records are bounded and deletion releases capacity`() {
        JobRegistry(retainedCapacity = 1, execute = { _, _ -> empty }).use { jobs ->
            val first = jobs.submit(submission)
            assertEquals(JobStatus.Succeeded, await(jobs, first.id).status)
            assertFailsWith<JobCapacityExceeded> { jobs.submit(submission) }
            assertTrue(jobs.delete(first.id))
            assertEquals(JobStatus.Succeeded, await(jobs, jobs.submit(submission).id).status)
        }
    }

    @Test
    fun `cooperative deadlines report timed out`() {
        JobRegistry(execute = { _, control ->
            while (true) control.checkpoint()
            @Suppress("UNREACHABLE_CODE")
            empty
        }).use { jobs ->
            val job = jobs.submit(submission.copy(timeoutMillis = 1))
            assertEquals(JobStatus.TimedOut, await(jobs, job.id).status)
        }
    }

    @Test
    fun `unexpected library errors fail the job and leave its worker usable`() {
        val calls = AtomicInteger()
        JobRegistry(workers = 1, execute = { _, _ ->
            if (calls.getAndIncrement() == 0) throw AssertionError("library invariant failed")
            empty
        }).use { jobs ->
            val failed = await(jobs, jobs.submit(submission).id)
            assertEquals(JobStatus.Failed, failed.status)
            assertEquals("execution_failed", failed.error?.code)
            assertEquals(JobStatus.Succeeded, await(jobs, jobs.submit(submission).id).status)
        }
    }

    @Test
    fun `shutdown cancels active work and rejects further submissions`() {
        val started = CountDownLatch(1)
        val jobs = JobRegistry(workers = 1, execute = { _, _ ->
            started.countDown()
            CountDownLatch(1).await()
            empty
        })
        try {
            val first = jobs.submit(submission)
            assertTrue(started.await(5, TimeUnit.SECONDS))
            val second = jobs.submit(submission)
            jobs.close()
            assertEquals(JobStatus.Cancelled, jobs.get(first.id)?.status)
            assertEquals(JobStatus.Cancelled, jobs.get(second.id)?.status)
            assertFailsWith<JobCapacityExceeded> { jobs.submit(submission) }
        } finally {
            jobs.close()
        }
    }

    private fun await(jobs: JobRegistry, id: String): JobSnapshot = runBlocking {
        withTimeout(5_000) {
            var snapshot: JobSnapshot
            do {
                delay(1)
                snapshot = checkNotNull(jobs.get(id))
            } while (!snapshot.status.terminal)
            snapshot
        }
    }
}
