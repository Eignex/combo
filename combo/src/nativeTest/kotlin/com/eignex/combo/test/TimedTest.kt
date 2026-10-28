package com.eignex.combo.test

import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.TimeSource

actual abstract class TimedTest actual constructor() {
    private var started = TimeSource.Monotonic.markNow()

    @BeforeTest
    fun startTimer() {
        started = TimeSource.Monotonic.markNow()
    }

    @AfterTest
    fun checkDuration() {
        val elapsed = started.elapsedNow()
        assertTrue(elapsed <= 300.milliseconds, "unit test took $elapsed; limit is 300ms")
    }
}
