package combo.bandit.glm

import com.eignex.kumulant.stat.summary.VarianceStat
import com.eignex.kumulant.stat.summary.WeightedVarianceResult

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
 * Combine two precision (inverse-variance) estimates that were tracked against different
 * sample means. Delegates to kumulant's [VarianceStat.merge], which implements the
 * Chan/Welford parallel-variance recurrence; precision and variance are reciprocals.
 *
 * Caller-side allocation cost (one stat per call) is fine for the import path, where
 * this fires once per import — not per coefficient.
 */
fun combinePrecision(p1: Float, p2: Float, m1: Float, m2: Float, n1: Float, n2: Float): Float {
    if (p1 == 0f || p2 == 0f) return 0f
    val stat = VarianceStat()
    stat.merge(WeightedVarianceResult(totalWeights = n1.toDouble(), mean = m1.toDouble(), variance = 1.0 / p1))
    stat.merge(WeightedVarianceResult(totalWeights = n2.toDouble(), mean = m2.toDouble(), variance = 1.0 / p2))
    val combined = stat.read()
    return (1.0 / combined.variance).toFloat()
}
