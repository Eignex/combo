package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context
import com.eignex.combo.test.TimedTest
import com.eignex.klause.solver.Sample
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class ObservationStoreTest : TimedTest() {
    @Test
    fun `snapshots retain insertion order and remain stable after appends`() {
        val store: ObservationStore = InMemoryObservationStore()
        val sample = BanditSample(Sample(booleanArrayOf(), longArrayOf()))
        val first = Observation(sample, Context.Empty, 3.0, 2.0)
        store.append(first)
        val snapshot = store.snapshot()
        store.append(Observation(sample, Context.Empty, 5.0))
        assertEquals(1L, snapshot.revision)
        assertEquals(listOf(first), snapshot.observations)
        assertEquals(listOf(3.0, 5.0), store.snapshot().observations.map { it.reward })
        assertEquals(2L, store.revision)
    }

    @Test
    fun `invalid observation weights and rewards are rejected`() {
        val sample = BanditSample(Sample(booleanArrayOf(), longArrayOf()))
        for (weight in listOf(-1.0, 0.0, Double.NaN, Double.POSITIVE_INFINITY)) {
            assertFailsWith<IllegalArgumentException> { Observation(sample, Context.Empty, 1.0, weight) }
        }
        assertFailsWith<IllegalArgumentException> { Observation(sample, Context.Empty, Double.NaN) }
    }
}
