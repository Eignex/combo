package com.eignex.combo.util

internal expect fun currentTimeNanos(): Long

internal fun checkedInt(value: Long): Int {
    if (value < Int.MIN_VALUE || value > Int.MAX_VALUE) throw ArithmeticException("integer overflow")
    return value.toInt()
}
