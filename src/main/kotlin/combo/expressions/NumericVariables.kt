package combo.expressions

import combo.model.Scope
import combo.model.VariableIndex
import combo.sat.*
import combo.util.*
import kotlin.math.max

/**
 * Use this to represent an integer in the model. This will use the least amount of bits required, so
 * always specify min/max values to reduce the size of the search space as much as possible.
 * @param min smallest allowed value (inclusive)
 * @param max largest allowed value (inclusive)
 */
class IntVar(
    name: String,
    override val optional: Boolean,
    override val parent: Value,
    val min: Int,
    val max: Int
) : Variable<Int, Int>(name) {

    init {
        require(max > min) { "Min should be greater than min." }
    }

    override val nbrValues: Int = let {
        val valueBits = max(Int.bitSize(max), Int.bitSize(min))
        val isSetBit = if (optional) 1 else 0
        val signedBit = if (min < 0) 1 else 0
        isSetBit + signedBit + valueBits
    }

    override fun rebase(parent: Value) =
        IntVar(name, optional, parent, min, max)

    override fun value(value: Int): IntLiteral {
        require(value in min..max)
        return IntLiteral(this, value)
    }

    fun isSigned() = min < 0

    override fun valueOf(
        instance: Instance,
        index: Int,
        parentLiteral: Int
    ): Int? {
        if ((parentLiteral != 0 && instance.literal(parentLiteral.toIx()) != parentLiteral) || (optional && !instance.isSet(
                index
            ))
        ) return null
        val offset = if (optional) 1 else 0
        return if (isSigned()) instance.getSignedInt(
            index + offset,
            nbrValues - offset
        ) else
            instance.getBits(index + offset, nbrValues - offset)
    }

    override fun implicitConstraints(
        scope: Scope,
        index: VariableIndex
    ): Sequence<Constraint> {
        val ix = index.valueIndexOf(this)
        val offset = if (optional) 1 else 0
        val zeros = IntRangeCollection(
            (ix + nbrValues - 1).toLiteral(false),
            (ix + offset).toLiteral(false)
        )
        return if (reifiedValue is Root) sequenceOf(
            IntBounds(
                ix + offset,
                min,
                max,
                nbrValues - offset
            )
        )
        else sequenceOf(
            ReifiedImplies(
                reifiedValue.not().toLiteral(index),
                Conjunction(zeros)
            ),
            ReifiedImplies(
                reifiedValue.toLiteral(index),
                IntBounds(ix + offset, min, max, nbrValues - offset)
            )
        )
    }

    override fun toString() = "IntVar($name in $min:$max)"

}

class IntLiteral(override val canonicalVariable: IntVar, val value: Int) :
    Literal {

    override val name: String get() = canonicalVariable.name

    override fun collectLiterals(index: VariableIndex, set: IntHashSet) {
        val ix = index.valueIndexOf(canonicalVariable)
        val offset = if (canonicalVariable.optional) {
            set.add(ix.toLiteral(true))
            1
        } else 0

        var k = value
        for (i in offset until canonicalVariable.nbrValues) {
            val kix = ix + i
            set.add(kix.toLiteral(k and 1 == 1))
            k = k ushr 1
        }
    }

    override fun toString() = "IntLiteral($name=$value)"
}

/**
 * Use this to represent a floating point number in the model. This will always use 32-33 bits.
 * Always specify min/max values to reduce the size of the search space as much as possible.
 * @param min smallest allowed value (inclusive)
 * @param max largest allowed value (inclusive)
 */
class FloatVar(
    name: String,
    override val optional: Boolean,
    override val parent: Value,
    val min: Float,
    val max: Float
) : Variable<Float, Float>(name) {

    init {
        require(max > min) { "Min should be greater than min." }
        // In javascript min/max are 64-bit so could be set to larger values
        require(max <= MAX_VALUE32) { "FloatVar overflow $name $max." }
        require(min >= -MAX_VALUE32) { "FloatVar overflow $name $min." }
    }

    override val nbrValues: Int = 32 + if (optional) 1 else 0

    override fun rebase(parent: Value) =
        FloatVar(name, optional, parent, min, max)

    override fun value(value: Float): FloatLiteral {
        require(value in min..max)
        return FloatLiteral(this, value)
    }

    override fun valueOf(
        instance: Instance,
        index: Int,
        parentLiteral: Int
    ): Float? {
        if ((parentLiteral != 0 && instance.literal(parentLiteral.toIx()) != parentLiteral) || (optional && !instance.isSet(
                index
            ))
        ) return null
        val offset = if (optional) 1 else 0
        return Float.fromBits(instance.getBits(index + offset, 32))
    }

    override fun implicitConstraints(
        scope: Scope,
        index: VariableIndex
    ): Sequence<Constraint> {
        val ix = index.valueIndexOf(this)
        val offset = if (optional) 1 else 0
        val zeros = IntRangeCollection(
            (ix + nbrValues - 1).toLiteral(false),
            (ix + offset).toLiteral(false)
        )
        return if (reifiedValue is Root) sequenceOf(
            FloatBounds(
                ix + offset,
                min,
                max
            )
        )
        else sequenceOf(
            ReifiedImplies(
                reifiedValue.not().toLiteral(index),
                Conjunction(zeros)
            ),
            ReifiedImplies(
                reifiedValue.toLiteral(index),
                FloatBounds(ix + offset, min, max)
            )
        )
    }

    override fun toString() = "FloatVar($name in $min:$max)"

}

class FloatLiteral(override val canonicalVariable: FloatVar, val value: Float) :
    Literal {

    override val name: String get() = canonicalVariable.name

    override fun collectLiterals(index: VariableIndex, set: IntHashSet) {
        val ix = index.valueIndexOf(canonicalVariable)
        val offset = if (canonicalVariable.optional) {
            set.add(ix.toLiteral(true))
            1
        } else 0

        var k = value.toRawBits()
        for (i in offset until canonicalVariable.nbrValues) {
            val kix = ix + i
            set.add(kix.toLiteral(k and 1 == 1))
            k = k ushr 1
        }
    }

    override fun toString() = "FloatLiteral($name=$value)"
}

class BitsVar constructor(
    name: String,
    override val optional: Boolean,
    override val parent: Value,
    val nbrBits: Int
) : Variable<Int, Instance>(name) {

    init {
        require(nbrBits > 0) { "nbrBits must be > 0." }
    }


    override val nbrValues: Int get() = nbrBits + if (optional) 1 else 0
    override fun rebase(parent: Value) =
        BitsVar(name, optional, parent, nbrBits)

    override fun valueOf(
        instance: Instance,
        index: Int,
        parentLiteral: Int
    ): Instance? {
        if ((parentLiteral != 0 && instance.literal(parentLiteral.toIx()) != parentLiteral) || (optional && !instance.isSet(
                index
            ))
        ) return null
        return BitArray(nbrBits).apply {
            var offset = if (optional) 1 else 0
            for (i in field.indices) {
                val nbrBits = if (i == field.lastIndex) nbrBits and 0x1F else 32
                field[i] = instance.getBits(index + offset, nbrBits)
                offset += nbrBits
            }
        }
    }

    /**
     * @param value is index of the bit field.
     */
    override fun value(value: Int) = BitValue(this, value)

    override fun implicitConstraints(
        scope: Scope,
        index: VariableIndex
    ): Sequence<Constraint> {
        if (reifiedValue is Root) return emptySequence()
        val ix = index.valueIndexOf(this)
        val offset = if (optional) 1 else 0
        val zeros = IntRangeCollection(
            (ix + nbrValues - 1).toLiteral(false),
            (ix + offset).toLiteral(false)
        )
        return sequenceOf(
            ReifiedImplies(
                reifiedValue.not().toLiteral(index),
                Conjunction(zeros)
            )
        )
    }

    override fun toString() = "BitsVar(nbrLiterals=$nbrBits)"
}

class BitValue constructor(
    override val canonicalVariable: BitsVar,
    val bitIndex: Int
) :
    Value {

    init {
        require(bitIndex in 0 until canonicalVariable.nbrBits) { "BitValue with index=$bitIndex is out of bound with $name." }
    }

    override val name: String get() = canonicalVariable.name
    override fun rebase(parent: Value) =
        (parent.canonicalVariable as BitsVar).value(bitIndex)

    override fun toLiteral(variableIndex: VariableIndex) =
        (variableIndex.valueIndexOf(canonicalVariable) + bitIndex + if (canonicalVariable.optional) 1 else 0).toLiteral(
            true
        )

    override fun toString() = "BitValue($name=$bitIndex)"
}
