package com.eignex.combo.bo

import com.eignex.combo.shared.WitnessBatch
import com.eignex.combo.shared.WitnessSearch
import com.eignex.combo.shared.WitnessTermination
import com.eignex.combo.shared.assignmentKey
import java.io.File
import kotlin.random.Random

/** Paired q=2 comparison; truth is used for evaluation only, never acquisition decisions. */
fun main(args: Array<String>) {
    val seeds = args.firstOrNull()?.takeIf { it.isNotBlank() }?.split(',')?.map(String::toInt) ?: (1501..1508).toList()
    val output = File("build/reports/joint-batch").also { it.mkdirs() }
    File(output, "outcomes.csv").writeText(
        "seed,sd,algorithm,calls,best_regret,recommendation_regret,model_fits,acquisition_evaluations\n",
    )
    File(output, "traces.csv").writeText("seed,sd,algorithm,call,round,observed,truth,x,y\n")
    for (seed in seeds) {
        for (sd in listOf(0.0, 0.03)) {
            for (algorithm in listOf("random", "exclusion", "mean-fantasy", "qei", "qnei")) {
                runJointBatch(seed, sd, algorithm, output)
            }
            println("joint-batch seed=$seed sd=$sd")
        }
    }
}

private fun runJointBatch(seed: Int, sd: Double, algorithm: String, output: File) {
    val problem = refinementProblem("noisy")
    val projection = SearchTransform(problem.space)
    val budget = 32
    var round = 0
    // Shared finite candidate pool per round, independent of model and acquisition random consumption.
    val pool = WitnessSearch { request, _ ->
        val random = Random(seed.toLong() * 1000003 + round)
        val excluded = request.exclude.map { it.assignmentKey() }.toSet()
        val points = List(16) { continuousPoint(doubleArrayOf(random.nextDouble(), random.nextDouble())).sample }
            .filter { it.assignmentKey() !in excluded }.take(request.count)
        WitnessBatch(
            points,
            if (points.size == request.count) WitnessTermination.CountReached else WitnessTermination.BudgetExhausted,
        )
    }
    val batchPolicy = when (algorithm) {
        "qei" -> JointBatchPolicy(
            JointExpectedImprovement(),
            JointPoolSearch(projection, pool, candidateCount = 16, starts = 2, passes = 1),
        )

        "qnei" -> JointBatchPolicy(
            JointNoisyExpectedImprovement(),
            JointPoolSearch(projection, pool, candidateCount = 16, starts = 2, passes = 1),
        )

        "mean-fantasy" -> SequentialBatchPolicy(FantasyPolicy.PosteriorMean)

        else -> SequentialBatchPolicy()
    }
    val optimizer = BayesianOptimizer(
        projection, GaussianProcessSurrogate(),
        acquisitionOptimizer = LocalAndRandomSearch(projection, pool, candidateCount = 16, localSteps = 0),
        batchPolicy = batchPolicy,
        initialDesign = LatinHypercubeInitialDesign(
            problem.space,
            budget,
            seed = seed,
            probability = if (algorithm == "random") 1.0 else 0.2,
            witnesses = noisyWitnesses(),
        ),
        outcome = OutcomeSpec(
            maximize = false,
            noise = if (sd > 0.0) ObservationNoise.Noisy else ObservationNoise.Deterministic,
            standardize = true,
        ),
        exploration = noisyWitnesses(), randomSeed = seed, repeatPolicy = RepeatPolicy.Never,
    )
    var best = Double.POSITIVE_INFINITY
    var evaluations = 0
    var calls = 0
    while (calls < budget) {
        round++
        val batch = optimizer.suggest(2)
        check(batch.size == 2)
        evaluations += optimizer.lastSuggestion!!.acquisitionEvaluations
        // Assign noise keys in creation order, then exercise out-of-order worker completion.
        val measurements = batch.mapIndexed { slot, trial ->
            noisyMeasurement(problem, trial.sample, seed, calls + slot, sd)
        }
        for (slot in (0 until batch.size).reversed()) {
            optimizer.complete(
                batch[slot].id,
                measurements[slot],
                noiseVariance = sd * sd,
            )
        }
        for ((slot, trial) in batch.withIndex()) {
            calls++
            val truth = problem.truth(trial.sample)
            best = minOf(best, truth)
            File(output, "traces.csv").appendText(
                "$seed,$sd,$algorithm,$calls,$round,${measurements[slot]},$truth," +
                    "${trial.sample.sample.reals.joinToString(",")}\n",
            )
        }
    }
    // All methods, including random sampling, use the same GP posterior-mean recommendation budget.
    val recommendation = optimizer.recommendOrThrow(noisySearch(problem))
    File(output, "outcomes.csv").appendText(
        "$seed,$sd,$algorithm,$calls,${best - 1.0},${problem.truth(recommendation) - 1.0}," +
            "${optimizer.fitCount},$evaluations\n",
    )
}
