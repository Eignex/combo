@file:JvmName("Sampling")

package combo.math

import combo.util.MIN_VALUE32
import kotlin.jvm.JvmName
import kotlin.math.*
import kotlin.random.Random

/**
 * Generate number in binary32 between from (inclusive) and until (inclusive).
 */
fun Random.nextFloat(from: Float, until: Float): Float {
    require(until > from)
    val size = until - from
    val r = if (size.isInfinite() && from.isFinite() && until.isFinite()) {
        val r1 = nextFloat() * (until / 2 - from / 2)
        from + r1 + r1
    } else from + nextFloat() * size
    return if (r > until) until.nextDown() else r
}

private fun Float.nextDown(): Float {
    return if (isNaN() || this == Float.NEGATIVE_INFINITY) this
    else {
        if (this == 0.0f) -MIN_VALUE32
        else Float.fromBits(toRawBits() + (if (this > 0.0f) -1 else +1))
    }
}

private fun Float.nextUp(): Float {
    return if (isNaN() || this == Float.POSITIVE_INFINITY) this
    else {
        if (this == 0.0f) MIN_VALUE32
        else Float.fromBits(toRawBits() + (if (this > 0.0f) +1 else -1))
    }
}

fun Random.nextFloatPos(): Float {
    while (true) {
        val u = nextFloat()
        if (u > 0.0f) return u
    }
}

fun Random.nextNormal(mean: Float = 0.0f, std: Float = 1.0f): Float {
    var u: Float
    var s: Float
    do {
        u = nextFloat() * 2 - 1
        val v = nextFloat() * 2 - 1
        s = u * u + v * v
    } while (s >= 1.0f || s == 0.0f)
    val mul = sqrt(-2.0f * ln(s) / s)
    return mean + std * u * mul
}

fun Random.nextLogNormal(mean: Float, variance: Float): Float {
    val phi = sqrt(variance + mean * mean)
    val mu = ln(mean * mean / phi)
    val sigma = sqrt(ln(phi * phi / (mean * mean)))
    return exp(nextNormal(mu, sigma))
}

fun Random.nextGamma(alpha: Float): Float {
    assert(alpha > 0)
    // Marsaglia and Tsang method
    if (alpha < 1.0) {
        val u = nextFloatPos()
        val g = nextGamma(1.0f + alpha) * u.pow(1.0f / alpha)
        return if (g == 0f) MIN_VALUE32 else g
    } else {
        val d = alpha - 1.0f / 3.0f
        val c = (1.0f / 3.0f) / sqrt(d)

        var v: Float
        var x: Float

        while (true) {
            do {
                x = nextNormal(0.0f, 1.0f)
                v = 1.0f + c * x
            } while (v <= 0)

            v = v * v * v
            val u = nextFloatPos()

            if (u < 1 - 0.0331f * x * x * x * x)
                break

            if (ln(u) < 0.5f * x * x + d * (1 - v + ln(v)))
                break
        }
        return d * v
    }
}

fun Random.nextBeta(alpha: Float, beta: Float): Float {
    val a = nextGamma(alpha)
    val b = nextGamma(beta)
    val m = max(a, b)
    val c = if (m == a + b) m.nextUp()
    else a + b
    return a / c
}

/**
 * Approximation implementation, it is only used in simulations.
 */
fun Random.nextPoisson(lambda: Float): Int {
    // Knuth's algorithm
    return if (lambda < 30) {
        val p = exp(-lambda)
        var n = 0
        var r = 1.0
        while (n < 1000 * lambda) {
            val u = nextFloat()
            r *= u
            if (r >= p) n++
            else return n
        }
        return n
    } else max(0.0f, nextNormal(lambda + .5f, sqrt(lambda))).toInt()
}

/**
 * Approximation implementation, it is only used in simulations.
 */
fun Random.nextBinomial(p: Float, n: Int = 1): Int {
    if (p > 0.5f) return n - nextBinomial(1 - p, n)
    val np = n * p
    val np1p = np * (1 - p)
    return if (n <= 5) {
        var sum = 0
        for (i in 0 until n)
            sum += if (nextFloat() < p) 1 else 0
        sum
    } else if (np < 20 && n > 100 && p < 0.05f)
        min(n, nextPoisson(np))
    else
        min(n, max(0.0f, nextNormal(np + 0.5f, sqrt(np1p))).toInt())
}

fun Random.nextGeometric(p: Float): Int {
    val u = nextFloatPos()
    return 1 + (ln1p(-u) / ln1p(-p)).toInt()
}

fun Random.nextExponential(rate: Float): Float {
    var u: Float
    do {
        u = nextFloat()
    } while (u == 0.0f || u == 1.0f)
    return -ln(u) / rate
}
