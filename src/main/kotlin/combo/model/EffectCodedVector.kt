package combo.model

import combo.expressions.Root
import combo.math.VectorView
import combo.math.vectors
import combo.sat.Instance
import combo.sat.literal
import combo.sat.toIx

/**
 * Vector that transforms 0-1 values to -1,0,1 depending on whether there are missing values.
 * This is used by regression based bandit algorithms like [combo.bandit.nn.NeuralLinearBandit] or
 * [combo.bandit.glm.LinearBandit].
 */
class EffectCodedVector(val model: Model, val instance: Instance) : VectorView {
    override val size: Int get() = instance.size
    override val sparse: Boolean get() = instance.sparse

    override fun iterator(): IntIterator {

        return object : IntIterator() {

            var varPtr = -1
            var valPtr = -1

            fun advance() {
                for (i in (varPtr + 1) until model.nbrVariables) {
                    val v = model.index.variable(i)
                    val lit = v.parentLiteral(model.index)
                    if (lit == 0 || lit == instance.literal(lit.toIx()) && v.nbrValues > 0) {
                        varPtr = i
                        valPtr = 0
                        return
                    }
                }
                varPtr = -1
                valPtr = -1
            }

            init {
                advance()
            }

            override fun hasNext() = varPtr >= 0

            override fun nextInt(): Int {
                assert(varPtr >= 0)
                assert(valPtr >= 0)
                val v = model.index.variable(varPtr)
                val ix = model.index.valueIndexOf(v)
                return if (valPtr == 0) {
                    val reifiedLiteral = if (v.reifiedValue is Root) 0
                    else v.reifiedValue.toLiteral(model.index)
                    if (v.nbrValues == 1 || (reifiedLiteral != 0 && reifiedLiteral != instance.literal(reifiedLiteral.toIx())))
                        advance()
                    else
                        valPtr++
                    ix
                } else {
                    val ret = ix + valPtr++
                    if (valPtr >= v.nbrValues)
                        advance()
                    ret
                }
            }
        }
    }

    override fun get(i: Int): Float {
        if (instance.isSet(i)) return 1f
        val reifiedLiteral = model.reifiedLiterals[i]
        return if (reifiedLiteral == 0 || instance.literal(reifiedLiteral.toIx()) == reifiedLiteral)
            -1f
        else 0f
    }

    override fun copy() = EffectCodedVector(model, instance.copy())
    override fun vectorCopy() = vectors.vector(toFloatArray())

    override fun toFloatArray() = FloatArray(size).also {
        for (i in iterator()) {
            it[i] = get(i)
        }
    }
}
