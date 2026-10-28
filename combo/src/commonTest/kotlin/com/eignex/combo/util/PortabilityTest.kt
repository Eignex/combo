package com.eignex.combo.util

import com.eignex.combo.test.TimedTest
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class PortabilityTest : TimedTest() {
    @Test
    fun `integer narrowing preserves boundaries and rejects overflow`() {
        for (value in listOf(Int.MIN_VALUE, -1, 0, 1, Int.MAX_VALUE)) {
            assertEquals(value, checkedInt(value.toLong()))
        }
        for (value in listOf(Long.MIN_VALUE, Int.MIN_VALUE.toLong() - 1, Int.MAX_VALUE.toLong() + 1, Long.MAX_VALUE)) {
            assertFailsWith<ArithmeticException> { checkedInt(value) }
        }
    }

    @Test
    fun `random sequence retains its seeded recurrence across platforms`() {
        for (seed in listOf(Int.MIN_VALUE, -1, 0, 1, Int.MAX_VALUE)) {
            val sequence = RandomSequence(seed)
            var state = seed
            repeat(20) {
                state = (1 + state) * 741103587
                val expected = Random(state)
                val actual = sequence.next()
                repeat(10) { assertEquals(expected.nextLong(), actual.nextLong()) }
            }
        }
    }

    @Test
    fun `default timestamps advance monotonically`() {
        var previous = currentTimeNanos()
        repeat(100) {
            val current = currentTimeNanos()
            assertTrue(current >= previous)
            previous = current
        }
    }
}
