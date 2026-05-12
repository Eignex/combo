package combo.math

import kotlin.math.absoluteValue
import kotlin.math.sqrt

/**
 * Perform cholesky decomposition downdate with vector x, such that A = L'*L - x*x'
 * The matrix is modified inline.
 * Ported from fortran dchdd.
 */
fun Matrix.choleskyDowndate(x: VectorView): Float {
    val L = this
    val p = x.size

    val s = vectors.zeroVector(p)
    val c = vectors.zeroVector(p)

    // Solve the system L.T*s = x
    s[0] = x[0] / L[0, 0]
    if (p > 1) {
        for (j in 1 until p) {
            var sum = 0f
            for (i in 0 until j)
                sum += L[i, j] * s[i]
            s[j] = x[j] - sum
            s[j] = s[j] / L[j, j]
        }
    }

    var norm = s.norm2()
    return if (norm > 0f && norm < 1f) {
        var alpha = sqrt(1 - norm * norm)
        for (ii in 0 until p) {
            val i = p - ii - 1
            val scale = alpha + s[i].absoluteValue
            val a = alpha / scale
            val b = s[i] / scale
            norm = sqrt(a * a + b * b)
            c[i] = a / norm
            s[i] = b / norm
            alpha = scale * norm
        }
        for (j in 0 until p) {
            var xx = 0f
            for (ii in 0..j) {
                val i = j - ii
                val t = c[i] * xx + s[i] * L[i, j]
                L[i, j] = c[i] * L[i, j] - s[i] * xx
                xx = t
            }
        }
        0f
    } else {
        norm
    }
}

fun Matrix.cholesky(): Matrix {
    val N = rows
    val L = vectors.zeroMatrix(N)
    for (i in 0 until N) {
        for (j in 0..i) {
            var sum = 0.0f
            for (k in 0 until j)
                sum += L[i, k] * L[j, k]

            if (i == j) {
                L[i, i] = sqrt(this[i, i] - sum)
            } else {
                L[i, j] = 1.0f / L[j, j] * (this[i, j] - sum)
            }
        }
        if (L[i, i] <= 0 || L[i, i].isNaN()) {
            L[i, i] = 1e-5f
        }
        // error("Matrix not positive definite")
    }
    return L
}
