package com.eignex.combo.util

import kotlin.time.TimeSource

private val origin = TimeSource.Monotonic.markNow()

internal actual fun currentTimeNanos(): Long = origin.elapsedNow().inWholeNanoseconds
