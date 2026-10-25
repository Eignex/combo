package com.eignex.server

import io.ktor.http.HttpStatusCode
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.runBlocking
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ResourceStoreTest {
    private class Counter : ManagedResource {
        override val type = "counter"
        var value = 0
        val closes = AtomicInteger()
        override fun close() { closes.incrementAndGet() }
    }

    @Test
    fun `resource creation happens once and repeated access preserves its state`() = runBlocking {
        OperationRunner().use { runner ->
            ResourceStore<Counter>(runner).use { store ->
                val creates = AtomicInteger()
                val handle = store.create { creates.incrementAndGet(); Counter() }
                coroutineScope {
                    (1..20).map {
                        async { store.access(handle.id) { resource, _ ->
                            val before = resource.value
                            Thread.yield()
                            resource.value = before + 1
                        } }
                    }.awaitAll()
                }
                assertEquals(1, creates.get())
                assertEquals(20, store.access(handle.id) { resource, _ -> resource.value })
            }
        }
    }

    @Test
    fun `waiting for one resource does not occupy the other worker`() = runBlocking {
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        OperationRunner(workers = 2).use { runner ->
            ResourceStore<Counter>(runner).use { store ->
                val first = store.create { Counter() }
                val second = store.create { Counter() }
                val busy = async(start = CoroutineStart.UNDISPATCHED) {
                    store.access(first.id) { _, _ ->
                        started.countDown()
                        assertTrue(release.await(5, TimeUnit.SECONDS))
                    }
                }
                try {
                    assertTrue(started.await(5, TimeUnit.SECONDS))
                    val waiting = async(start = CoroutineStart.UNDISPATCHED) { store.access(first.id) { _, _ -> 1 } }
                    assertEquals(2, store.access(second.id, timeoutMillis = 1000) { _, _ -> 2 })
                    release.countDown()
                    busy.await()
                    assertEquals(1, waiting.await())
                } finally {
                    release.countDown()
                }
            }
        }
    }

    @Test
    fun `admission is bounded and completing calls releases capacity`() = runBlocking {
        val started = CountDownLatch(1)
        val release = CountDownLatch(1)
        OperationRunner(workers = 1, queueCapacity = 1).use { runner ->
            val first = async(start = CoroutineStart.UNDISPATCHED) {
                runner.run { _ -> started.countDown(); assertTrue(release.await(5, TimeUnit.SECONDS)); 1 }
            }
            try {
                assertTrue(started.await(5, TimeUnit.SECONDS))
                val second = async(start = CoroutineStart.UNDISPATCHED) { runner.run { 2 } }
                val error = assertFailsWith<ApiException> { runner.run { 3 } }
                assertEquals(HttpStatusCode.ServiceUnavailable, error.status)
                release.countDown()
                assertEquals(1, first.await())
                assertEquals(2, second.await())
                assertEquals(3, runner.run { 3 })
            } finally {
                release.countDown()
            }
        }
    }

    @Test
    fun `cooperative deadlines cancel computation without poisoning the worker`() = runBlocking {
        OperationRunner(workers = 1).use { runner ->
            assertFailsWith<TimeoutCancellationException> {
                runner.run(timeoutMillis = 20) { control ->
                    while (true) control.checkpoint()
                }
            }
            assertEquals(7, runner.run { 7 })
        }
    }

    @Test
    fun `deletion closes a resource once and releases retained capacity`() = runBlocking {
        OperationRunner().use { runner ->
            val counter = Counter()
            ResourceStore<Counter>(runner, capacity = 1).use { store ->
                val handle = store.create { counter }
                assertFailsWith<ApiException> { store.create { Counter() } }
                store.delete(handle.id)
                assertEquals(1, counter.closes.get())
                assertEquals(HttpStatusCode.NotFound, assertFailsWith<ApiException> {
                    store.access(handle.id) { _, _ -> Unit }
                }.status)
                val replacement = store.create { Counter() }
                assertEquals(listOf(replacement), store.list())
            }
            assertEquals(1, counter.closes.get())
        }
    }

    @Test
    fun `failed construction does not consume resource capacity`() = runBlocking {
        OperationRunner().use { runner ->
            ResourceStore<Counter>(runner, capacity = 1).use { store ->
                assertFailsWith<IllegalArgumentException> { store.create { throw IllegalArgumentException("bad schema") } }
                val resource = store.create { Counter() }
                assertEquals(listOf(resource), store.list())
            }
        }
    }

    @Test
    fun `shutdown cancels computation and closes its resource after the call exits`() = runBlocking<Unit> {
        val started = CountDownLatch(1)
        val runner = OperationRunner()
        val store = ResourceStore<Counter>(runner)
        val counter = Counter()
        try {
            val handle = store.create { counter }
            val active = async(start = CoroutineStart.UNDISPATCHED) {
                store.access(handle.id) { _, control ->
                    started.countDown()
                    while (true) control.checkpoint()
                }
            }
            assertTrue(started.await(5, TimeUnit.SECONDS))
            store.close()
            assertEquals(0, counter.closes.get())
            runner.close()
            assertFailsWith<CancellationException> { active.await() }
            assertEquals(1, counter.closes.get())
            assertFailsWith<ApiException> { runner.run { 1 } }
        } finally {
            store.close()
            runner.close()
        }
    }
}
