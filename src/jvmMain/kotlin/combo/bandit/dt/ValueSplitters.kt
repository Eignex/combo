package combo.bandit.dt

import combo.expressions.BitsVar
import combo.expressions.Flag
import combo.expressions.FloatVar
import combo.expressions.IntVar
import combo.expressions.Select
import combo.expressions.Variable
import combo.math.permutation
import combo.model.*
import combo.sat.not
import combo.sat.toBoolean
import combo.sat.toIx
import combo.sat.toLiteral
import combo.util.IntCollection
import combo.util.bsr
import kotlin.random.Random

fun defaultValueSplitter(model: Model, variable: Variable<*, *>): ValueSplitter {
    val index = model.index.valueIndexOf(variable)
    val parentLiteral = variable.parentLiteral(model.index)
    return when (variable) {
        is Flag -> FlagSplitter(variable, index, parentLiteral)
        is IntVar -> IntValueSplitter(variable, index, parentLiteral)
        is FloatVar -> FloatValueSplitter(variable, index, parentLiteral)
        is BitsVar -> StandardValueSplitter(variable, index, parentLiteral)
        is Select<*, *> -> StandardValueSplitter(variable, index, parentLiteral)
        else -> throw UnsupportedOperationException("Add custom splitter for variable $variable.")
    }
}

interface ValueSplitter {
    fun nextSplit(setLiterals: IntCollection, auditedValues: IntCollection, rng: Random): Int
}

class FlagSplitter(val variable: Flag<*>, val index: Int, val parentLiteral: Int) : ValueSplitter {
    override fun nextSplit(setLiterals: IntCollection, auditedValues: IntCollection, rng: Random): Int {
        if (index in auditedValues || index.toLiteral(true) in setLiterals || index.toLiteral(false) in setLiterals)
            return -1
        if (parentLiteral != 0) {
            if (!parentLiteral in setLiterals)
                return -1
            else if (parentLiteral !in setLiterals && parentLiteral.toIx() !in auditedValues)
                return parentLiteral.toIx()
        }
        return index
    }
}

open class StandardValueSplitter(val variable: Variable<*, *>, val index: Int, val parentLiteral: Int) : ValueSplitter {
    override fun nextSplit(setLiterals: IntCollection, auditedValues: IntCollection, rng: Random): Int {
        var returnReified = false
        if (parentLiteral != 0) {
            if (!parentLiteral in setLiterals)
                return -1
            else if (parentLiteral !in setLiterals && parentLiteral.toIx() !in auditedValues)
                returnReified = true
        }

        val offset = if (variable.optional) {
            val lit = index.toLiteral(true)
            if (!lit in setLiterals) return -1
            else if (!returnReified && lit !in setLiterals && index !in auditedValues) return index
            1
        } else 0
        if (returnReified) return parentLiteral.toIx()

        for (i in permutation(variable.nbrValues - offset, rng)) {
            val ix = i + index + offset
            if (ix !in auditedValues && ix.toLiteral(true) !in setLiterals && ix.toLiteral(false) !in setLiterals)
                return ix
            else continue
        }
        return -1
    }
}

class IntValueSplitter(val variable: IntVar, val index: Int, val parentLiteral: Int) : ValueSplitter {
    override fun nextSplit(setLiterals: IntCollection, auditedValues: IntCollection, rng: Random): Int {
        var returnReified = false
        if (parentLiteral != 0) {
            if (!parentLiteral in setLiterals)
                return -1
            else if (parentLiteral !in setLiterals && parentLiteral.toIx() !in auditedValues)
                returnReified = true
        }
        val offset = if (variable.optional) {
            val lit = index.toLiteral(true)
            if (!lit in setLiterals) return -1
            else if (!returnReified && lit !in setLiterals && index !in auditedValues) return index
            1
        } else 0
        if (returnReified) return parentLiteral.toIx()

        // TODO handle contraditions
        var unset = -1 ushr (32 - variable.nbrValues + offset)
        //var unset = if (variable.isSigned()) -1 ushr (32 - variable.nbrValues + offset)
        //else -1 ushr (32 - (variable.nbrValues -offset (Int.bsr(variable.min xor variable.max) + 1))

        for (lit in setLiterals) {
            val ix = lit.toIx() - index - offset
            if (ix in 0..31) {
                //if (((unset ushr ix) and 1) == 0 &&
                //((variable.min ushr ix) and 1) != (if (lit.toBoolean()) 1 else 0))
                //return -1 // Contradiction, the forced min/max bit does not equal v
                unset = unset and (1 shl ix).inv()
            }
        }
        for (v in auditedValues) {
            val ix = v - index - offset
            if (ix in 0..31)
                unset = unset and (1 shl ix).inv()
        }
        return if (unset != 0) index + offset + Int.bsr(unset)
        else -1
    }
}

class FloatValueSplitter(val variable: FloatVar, val index: Int, val parentLiteral: Int) : ValueSplitter {
    override fun nextSplit(setLiterals: IntCollection, auditedValues: IntCollection, rng: Random): Int {
        var returnReified = false
        if (parentLiteral != 0) {
            if (!parentLiteral in setLiterals)
                return -1
            else if (parentLiteral !in setLiterals && parentLiteral.toIx() !in auditedValues)
                returnReified = true
        }
        val offset = if (variable.optional) {
            val lit = index.toLiteral(true)
            if (!lit in setLiterals) return -1
            else if (!returnReified && lit !in setLiterals && index !in auditedValues) return index
            1
        } else 0
        if (returnReified) return parentLiteral.toIx()
        // All bits that are fixed will be zero
        var unset = -1 ushr (32 - (Int.bsr(variable.min.toRawBits() xor variable.max.toRawBits()) + 1))

        for (lit in setLiterals) {
            val ix = lit.toIx() - index - offset
            if (ix in 0..31) {
                if (((unset ushr ix) and 1) == 0 &&
                        ((variable.min.toRawBits() ushr ix) and 1) != (if (lit.toBoolean()) 1 else 0))
                    return -1 // Contradiction, the forced min/max bit does not equal v
                unset = unset and (1 shl ix).inv()
            }
        }
        for (v in auditedValues) {
            val ix = v - index - offset
            if (ix in 0..31)
                unset = unset and (1 shl ix).inv()
        }

        return if (unset == 0) -1
        else index + offset + Int.bsr(unset)
    }
}
