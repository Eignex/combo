package combo.math

import kotlin.math.absoluteValue

/**
 * Returns the joint greatest common divisor among int numbers (including negative).
 */
fun gcdAll(vararg ns: Int): Int {
    if (ns.isEmpty()) return 0
    if (ns.size == 1) return ns[0]
    var g = gcd(ns[0].absoluteValue, ns[1].absoluteValue)
    if (g == 1) return 1
    for (i in 2 until ns.size) {
        g = gcd(g, ns[i].absoluteValue)
        if (g == 1) return 1
    }
    return g
}

/**
 * Returns the greatest common divisor among positive numbers. Implemented with Stein's algorithm.
 */
fun gcd(a: Int, b: Int): Int {
    if (a == 0) return b
    if (b == 0) return a

    var p = a
    var q = b

    var n = 0
    while ((p or q) and 1 == 0) {
        p = p shr 1
        q = q shr 1
        n++
    }

    while ((p and 1) == 0)
        p = p shr 1

    do {
        while ((q and 1) == 0)
            q = q shr 1

        if (p > q) {
            val temp = p
            p = q
            q = temp
        }

        q = (q - p)
    } while (q != 0)

    return p shl n
}
