package combo.bandit.glm

/**
 * Welford-style helpers used by the linear models when blending an imported checkpoint
 * with the current model state. They live next to the linear models because that's the
 * only consumer; if more callers appear, lift to a shared location.
 */

/** Weighted combination of two means. Returns `0f` if both weights are zero. */
fun combineMean(m1: Float, m2: Float, n1: Float, n2: Float): Float {
    val n = n1 + n2
    return if (n == 0f) 0f else (m1 * n1 + m2 * n2) / n
}

/**
 * Combine two precision (inverse-variance) estimates that were tracked against
 * different sample means. Mirrors the Chan/Welford parallel-variance recurrence,
 * expressed in precision space.
 */
fun combinePrecision(p1: Float, p2: Float, m1: Float, m2: Float, n1: Float, n2: Float): Float {
    val n = n1 + n2
    return if (p1 == 0f || p2 == 0f) 0f
    else n * n / (n1 * n2 * (m1 - m2) * (m1 - m2) + n * (n1 / p1 + n2 / p2))
}
