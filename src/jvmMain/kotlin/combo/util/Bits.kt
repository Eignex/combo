@file:JvmName("Bits")

package combo.util

import kotlin.jvm.JvmName
import kotlin.math.absoluteValue

/**
 * Closest power of 2, e.g. power2(5) = 8, power2(9) = 16
 */
fun Int.Companion.power2(value: Int): Int {
    var x = value - 1
    x = x or (x shr 1)
    x = x or (x shr 2)
    x = x or (x shr 4)
    x = x or (x shr 8)
    x = x or (x shr 16)
    return x + 1
}

private val DE_BRUIJN_POSITION_MSB = intArrayOf(
    0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30,
    8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31
)

/**
 * Bit scan reverse for the most significant bit set.
 */
fun Int.Companion.bsr(value: Int): Int {
    var x = value
    x = x or (x ushr 1)
    x = x or (x ushr 2)
    x = x or (x ushr 4)
    x = x or (x ushr 8)
    x = x or (x ushr 16)
    val i = 0x07C4ACDD
    val i1 = (x * i) ushr 27
    return DE_BRUIJN_POSITION_MSB[i1]
}

private val DE_BRUIJN_POSITION_LSB = intArrayOf(
    0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
    31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
)

/**
 * Bit scan forward for the least significant bit set.
 */
fun Int.Companion.bsf(value: Int): Int {
    return DE_BRUIJN_POSITION_LSB[((value and -value) * 0x077CB531) ushr 27]
}

fun Int.Companion.bitSize(value: Int): Int {
    return (
        if (value < 0) {
            Int.bsr(value.absoluteValue - 1)
        } else {
            Int.bsr(value)
        }
        ) + 1
}

/**
 * This is adapted to kotlin from the JDK java.lang.Integer.bitCount method.
 */
fun Int.Companion.bitCount(value: Int): Int {
    var i = value
    i = i - ((i ushr 1) and 0x55555555)
    i = (i and 0x33333333) + ((i ushr 2) and 0x33333333)
    i = (i + (i ushr 4)) and 0x0F0F0F0F
    i = i + (i ushr 8)
    i = i + (i ushr 16)
    return i and 0x3F
}

/**
 * Maximum value of Float in binary32 format. This ensures that the max value is same in all platforms.
 */
val MAX_VALUE32 = 3.4028235e+38f
val MIN_VALUE32 = 1.4e-45f
