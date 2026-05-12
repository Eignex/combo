package combo.bandit.dt

import combo.bandit.glm.ConstantRate
import combo.bandit.glm.LearningRateSchedule
import combo.math.VectorView
import combo.math.nextGamma
import combo.math.nextNormal
import combo.math.vectors
import kotlin.random.Random

interface VoteStrategy {
    fun weights(votesYes: ShortArray, votesNo: ShortArray, rng: Random, trees: Int, t: Long): VectorView
}

class BetaRandomizedVotes(val priorVotes: Float = 1f, val explorationRate: LearningRateSchedule) : VoteStrategy {
    override fun weights(votesYes: ShortArray, votesNo: ShortArray, rng: Random, trees: Int, t: Long): VectorView {
        val lr = explorationRate.rate(t)
        return if (lr <= 1e-10f) {
            SumVotes(priorVotes, 0f).weights(votesYes, votesNo, rng, trees, t)
        } else {
            val weights = FloatArray(votesYes.size) {
                if (votesYes[it] == votesNo[it]) 0f
                else {
                    val a = votesYes[it]
                    val b = votesNo[it]
                    val ahat = rng.nextGamma(priorVotes + a / trees / lr)
                    val bhat = rng.nextGamma(priorVotes + b / trees / lr)
                    val p = ahat / (ahat + bhat)
                    val res = 2 * p - 1
                    res
                }
            }
            vectors.vector(weights)
        }
    }
}

class SumVotes(val priorVotes: Float = 1f, val exploration: Float = 0.01f, val explorationRate: LearningRateSchedule = ConstantRate(1f)) : VoteStrategy {
    override fun weights(votesYes: ShortArray, votesNo: ShortArray, rng: Random, trees: Int, t: Long): VectorView {
        val e = exploration * explorationRate.rate(t)
        val weights = FloatArray(votesYes.size) {
            val a = votesYes[it] + priorVotes
            val b = votesNo[it] + priorVotes
            val p = (a - b) / trees
            rng.nextNormal(p, e)
        }
        return vectors.vector(weights)
    }
}
