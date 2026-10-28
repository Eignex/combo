package com.eignex.combo.shared

import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.Sample

/** Preserve affine equalities while proposing between witnesses with identical discrete coordinates. */
internal fun interpolateWitness(left: Sample, right: Sample, fraction: Double): Sample? {
    require(fraction in 0.0..1.0)
    if (!left.bools.contentEquals(right.bools) || !left.ints.contentEquals(right.ints)) return null
    val a = left.exactReals ?: left.reals.map { BigFraction.ofDouble(it) ?: return null }
    val b = right.exactReals ?: right.reals.map { BigFraction.ofDouble(it) ?: return null }
    if (a.size != b.size) return null
    val weight = BigFraction.ofDouble(fraction) ?: return null
    val exact = a.indices.map { a[it] * (BigFraction.ONE - weight) + b[it] * weight }
    val coordinates = DoubleArray(exact.size) { index ->
        val projected = exact[index].toDouble()
        if (!projected.isFinite() || (projected == 0.0 && !exact[index].isZero)) return null
        projected
    }
    return Sample(left.bools, left.ints, coordinates, exactReals = exact, exactInts = left.exactInts)
}
