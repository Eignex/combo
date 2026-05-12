package combo.bandit.dt

import combo.math.VarianceEstimator
import combo.math.chi2CdfDf1
import combo.math.fCdfDf1
import kotlin.math.log2
import kotlin.math.sqrt


interface SplitMetric {

    fun split(total: VarianceEstimator, pos: Array<VarianceEstimator>, neg: Array<VarianceEstimator>,
              minSamplesSplit: Float, minSamplesLeaf: Float): SplitInfo {
        var top1 = 0.0f
        var top2 = 0.0f
        var bestI = -1

        val tv = totalValue(total)

        for (i in pos.indices) {
            val nPos = pos[i].nbrWeightedSamples
            val nNeg = neg[i].nbrWeightedSamples
            if (nPos < minSamplesLeaf || nNeg < minSamplesLeaf || nPos + nNeg < minSamplesSplit) continue
            val v = tv - value(total, pos[i], neg[i])
            if (v > top1) {
                bestI = i
                top2 = top1
                top1 = v
            } else if (v > top2)
                top2 = v
        }

        return SplitInfo(top1, top2, bestI)
    }

    fun totalValue(total: VarianceEstimator): Float = 0.0f
    fun value(total: VarianceEstimator, pos: VarianceEstimator, neg: VarianceEstimator): Float
}

data class SplitInfo(val top1: Float, val top2: Float, val i: Int)

object VarianceReduction : SplitMetric {
    override fun totalValue(total: VarianceEstimator) = total.variance
    override fun value(total: VarianceEstimator, pos: VarianceEstimator, neg: VarianceEstimator): Float {
        val nPos = pos.nbrWeightedSamples
        val nNeg = neg.nbrWeightedSamples
        val n = nPos + nNeg
        return (nNeg / n) * neg.variance + (nPos / n) * pos.variance
    }
}

object TTest : SplitMetric {
    override fun totalValue(total: VarianceEstimator) = 1.0f
    override fun value(total: VarianceEstimator, pos: VarianceEstimator, neg: VarianceEstimator): Float {
        val nPos = pos.nbrWeightedSamples
        val nNeg = neg.nbrWeightedSamples
        val n = nPos + nNeg
        val diff = pos.mean - neg.mean
        val t = diff / sqrt((pos.variance / nPos + neg.variance / nNeg))
        val F = t * t
        return 1 - fCdfDf1(F, n - 1)
    }
}

object EntropyReduction : SplitMetric {
    override fun totalValue(total: VarianceEstimator): Float {
        val n = total.nbrWeightedSamples
        val pos = total.sum
        val neg = n - total.sum
        val rp = pos / n
        val rn = neg / n
        return -rp * log2(rp) - rn * log2(rn)
    }

    override fun value(total: VarianceEstimator, pos: VarianceEstimator, neg: VarianceEstimator): Float {
        val nPos = pos.nbrWeightedSamples
        val nNeg = neg.nbrWeightedSamples
        val n = nPos + nNeg
        val ePos = totalValue(pos)
        val eNeg = totalValue(neg)
        return ePos * nPos / n + eNeg * nNeg / n
    }
}

object ChiSquareTest : SplitMetric {
    override fun totalValue(total: VarianceEstimator) = 1.0f
    override fun value(total: VarianceEstimator, pos: VarianceEstimator, neg: VarianceEstimator): Float {
        val nPos = pos.nbrWeightedSamples
        val nNeg = neg.nbrWeightedSamples
        val n = nPos + nNeg
        val actualPN = nPos - pos.sum
        val actualPP = pos.sum
        val actualNN = nNeg - neg.sum
        val actualNP = neg.sum

        val totalP = pos.sum + neg.sum
        val totalN = n - totalP
        val expectedPN = nPos * totalN / n
        val expectedPP = nPos * totalP / n
        val expectedNN = nNeg * totalN / n
        val expectedNP = nNeg * totalP / n

        val diffPN = actualPN - expectedPN
        val diffPP = actualPP - expectedPP
        val diffNN = actualNN - expectedNN
        val diffNP = actualNP - expectedNP

        val xPN = diffPN * diffPN / expectedPN
        val xPP = diffPP * diffPP / expectedPP
        val xNN = diffNN * diffNN / expectedNN
        val xNP = diffNP * diffNP / expectedNP

        val x = xPN + xPP + xNN + xNP
        return 1 - chi2CdfDf1(x)
    }
}

object GiniCoefficient : SplitMetric {
    override fun totalValue(total: VarianceEstimator): Float {
        val n = total.nbrWeightedSamples
        val pos = total.sum
        val neg = n - total.sum
        val rp = pos / n
        val rn = neg / n
        return -rp * rp - rn * rn
    }

    override fun value(total: VarianceEstimator, pos: VarianceEstimator, neg: VarianceEstimator): Float {
        val nPos = pos.nbrWeightedSamples
        val nNeg = neg.nbrWeightedSamples
        val n = nPos + nNeg
        val gPos = totalValue(pos)
        val gNeg = totalValue(neg)
        return gPos * nPos / n + gNeg * nNeg / n
    }
}

