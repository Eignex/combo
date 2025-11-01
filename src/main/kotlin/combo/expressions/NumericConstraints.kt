package combo.expressions

import combo.math.nextFloat
import combo.sat.*
import combo.util.IntRangeCollection
import combo.util.bitCount
import kotlin.random.Random

sealed class NumericConstraint(override val literals: IntRangeCollection) :
    Constraint {
    override val priority: Int get() = 300
    override fun cacheUpdate(cacheResult: Int, newLit: Int) = 0
    override fun cache(instance: Instance) = 0
    override fun isUnit() = false
    override fun unitPropagation(unit: Int) = this
}

class IntBounds(literals: IntRangeCollection, val min: Int, val max: Int) :
    NumericConstraint(literals) {

    constructor(ix: Int, min: Int, max: Int, nbrLiterals: Int) :
            this(
                IntRangeCollection(
                    ix.toLiteral(true),
                    (ix + nbrLiterals - 1).toLiteral(true)
                ), min, max
            )

    private fun isSigned() = min < 0 || max < 0

    override fun violations(instance: Instance, cacheResult: Int): Int {
        val value = if (isSigned()) instance.getSignedInt(
            literals.min.toIx(),
            literals.size
        )
        else instance.getBits(literals.min.toIx(), literals.size)
        val coercedInt = value.coerceIn(min, max)
        val changedBits = coercedInt xor value
        return Int.bitCount(changedBits)
    }

    override fun coerce(instance: Instance, rng: Random) {
        val coerced =
            rng.nextInt(min, if (max == Int.MAX_VALUE) max else max + 1)
        if (isSigned()) instance.setSignedInt(
            literals.min.toIx(),
            literals.size,
            coerced
        )
        else instance.setBits(literals.min.toIx(), literals.size, coerced)
    }

    override fun toString() =
        "IntBounds(${literals.min.toIx()}..${literals.max.toIx()} in $min:$max)"
}

class FloatBounds(
    literals: IntRangeCollection,
    val min: Float,
    val max: Float
) : NumericConstraint(literals) {

    init {
        assert(min.isFinite())
        assert(max.isFinite())
        assert(max > min)
    }

    constructor(
        ix: Int,
        min: Float,
        max: Float
    ) : this(
        IntRangeCollection(ix.toLiteral(true), (ix + 31).toLiteral(true)),
        min,
        max
    )

    private fun Float.coerceIn(
        minimumValue: Float,
        maximumValue: Float
    ): Float {
        // We use the compareTo here rather than the '<' and '>' operators to make sure that -0.0f < 0.0f
        if (this.compareTo(minimumValue) < 0) return minimumValue
        if (this.compareTo(maximumValue) > 0) return maximumValue
        return this
    }

    override fun violations(instance: Instance, cacheResult: Int): Int {
        val value = instance.getFloat(literals.min.toIx())
        val coercedBits = value.coerceIn(min, max).toRawBits()
        val valueBits = value.toRawBits()
        val changedBits = coercedBits xor valueBits
        return Int.bitCount(changedBits)
    }

    override fun coerce(instance: Instance, rng: Random) {
        val coerced = rng.nextFloat(min, max)
        instance.setFloat(literals.min.toIx(), coerced)
    }

    override fun toString() =
        "FloatBounds(${literals.min.toIx()}..${literals.max.toIx()} in $min:$max)"
}
