package com.eignex.combo.shared

import com.eignex.klause.ir.Problem
import com.eignex.klause.solver.objective.LinearObjective
import kotlin.math.abs
import kotlin.math.roundToLong

/**
 * Translate model scores to Klause's integer objective. A shared positive scale
 * preserves relative weights to about nine decimal digits; values below that
 * precision can tie. Constants are omitted because they cannot change the winner.
 * Domain bounds cap the total magnitude, leaving headroom for solver arithmetic.
 */
internal fun integerObjective(
    problem: Problem,
    boolWeights: DoubleArray,
    intCoefficients: DoubleArray = DoubleArray(problem.numIntVars),
    realCoefficients: DoubleArray = DoubleArray(problem.numRealVars),
): LinearObjective {
    require(boolWeights.size == problem.numBoolVars)
    require(intCoefficients.size == problem.numIntVars)
    require(realCoefficients.size == problem.numRealVars)
    require(
        boolWeights.all { it.isFinite() } && intCoefficients.all { it.isFinite() } &&
            realCoefficients.all { it.isFinite() },
    ) {
        "objective coefficients must be finite"
    }
    val largest = maxOf(
        boolWeights.maxOfOrNull { abs(it) } ?: 0.0,
        intCoefficients.maxOfOrNull { abs(it) } ?: 0.0,
        realCoefficients.maxOfOrNull { abs(it) } ?: 0.0,
    )
    if (largest == 0.0) {
        return LinearObjective(
            LongArray(boolWeights.size),
            LongArray(intCoefficients.size),
            realCoefficients = realCoefficients,
        )
    }
    var magnitude = boolWeights.sumOf { abs(it / largest) }
    var roundingMagnitude = boolWeights.size.toDouble()
    for (i in intCoefficients.indices) {
        if (intCoefficients[i] == 0.0) continue
        val bounds = problem.intBounds
        require(bounds.hasLower(i) && bounds.hasUpper(i)) { "bandit objectives require bounded integers" }
        val extent = maxOf(abs(bounds.lower(i).toDouble()), abs(bounds.upper(i).toDouble()))
        magnitude += abs(intCoefficients[i] / largest) * extent
        roundingMagnitude += extent
    }
    val budget = (1L shl 50).toDouble()
    require(roundingMagnitude < budget) { "integer domains exceed the objective precision budget" }
    val scale = if (magnitude == 0.0) 1.0 else minOf(1e9, (budget - roundingMagnitude) / magnitude)
    return LinearObjective(
        LongArray(boolWeights.size) { (boolWeights[it] / largest * scale).roundToLong() },
        LongArray(intCoefficients.size) { (intCoefficients[it] / largest * scale).roundToLong() },
        realCoefficients = DoubleArray(realCoefficients.size) { realCoefficients[it] / largest * scale },
    )
}
