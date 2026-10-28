package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context
import com.eignex.koblas.DenseVector
import com.eignex.koblas.Vector

/** One objective evaluation, optionally with a known noiseless reference for benchmark regret. */
data class BenchmarkEvaluation(
    /** Response delivered to the optimizer. */
    val reward: Double,
    /** Positive simulated evaluation cost. */
    val cost: Double = 1.0,
    /** Known true objective value in synthetic benchmarks; defaults to the observed response. */
    val reference: Double = reward,
) {
    init {
        require(reward.isFinite() && reference.isFinite() && cost > 0.0 && cost.isFinite())
    }
}

/** Per-evaluation progress in original objective units, with numerical storage supplied by Koblas. */
class BenchmarkResult(
    /** Best evaluated reference value after each completion. */
    val bestValues: Vector,
    /** Cumulative external evaluation cost. */
    val cumulativeCost: Vector,
    /** Simple regret against a supplied known optimum; null when no optimum is known. */
    val simpleRegret: Vector?,
)

/** Reproducible study harness; seed the supplied policy and objective noise source explicitly. */
object OptimizationBenchmark {
    /**
     * Run a fresh optimizer for at most budget objective evaluations in batches. An empty suggestion
     * batch stops the run; partial batches are evaluated before continuing.
     * Reference regret uses evaluated points, rather than untested recommendations.
     */
    fun run(
        optimizer: TrialOptimizer,
        budget: Int,
        evaluate: (BanditSample) -> BenchmarkEvaluation,
        context: Context = Context.Empty,
        batchSize: Int = 1,
        outcome: OutcomeSpec = OutcomeSpec(),
        optimum: Double? = null,
    ): BenchmarkResult {
        require(budget >= 0 && batchSize > 0 && (optimum == null || optimum.isFinite()))
        val best = mutableListOf<Double>()
        val costs = mutableListOf<Double>()
        var current: Double? = null
        var totalCost = 0.0
        while (best.size < budget) {
            val batch = optimizer.suggest(minOf(batchSize, budget - best.size), context)
            if (batch.size == 0) break
            for (trial in batch) {
                val value = evaluate(trial.sample)
                optimizer.complete(trial.id, value.reward)
                if (current == null || outcome.score(
                        value.reference,
                    ) > outcome.score(current)
                ) {
                    current = value.reference
                }
                totalCost += value.cost
                best += requireNotNull(current)
                costs += totalCost
            }
        }
        return BenchmarkResult(
            DenseVector.of(best.toDoubleArray()),
            DenseVector.of(costs.toDoubleArray()),
            optimum?.let { target ->
                DenseVector.of(
                    DoubleArray(best.size) { maxOf(0.0, outcome.score(target) - outcome.score(best[it])) },
                )
            },
        )
    }
}
