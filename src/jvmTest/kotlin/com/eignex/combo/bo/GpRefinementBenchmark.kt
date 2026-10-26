package com.eignex.combo.bo

import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.shared.WitnessBatch
import com.eignex.combo.shared.WitnessSearch
import com.eignex.combo.shared.WitnessTermination
import com.eignex.combo.shared.assignmentKey
import com.eignex.kumulant.stat.score.MaeLossStat
import com.eignex.kumulant.stat.summary.MeanStat
import java.io.File
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.pow
import kotlin.random.Random

private class TwoDimensions : DecisionSpace() {
    val x by floatVar(0.0, 1.0)
    val y by floatVar(0.0, 1.0)
}
private class ThreeDimensions : DecisionSpace() {
    val x by floatVar(
        0.0,
        1.0,
    )
    val y by floatVar(0.0, 1.0)
    val z by floatVar(0.0, 1.0)
}
private class SixDimensions : DecisionSpace() {
    val a by floatVar(0.0, 1.0)
    val b by floatVar(0.0, 1.0)
    val c by floatVar(0.0, 1.0)
    val d by floatVar(0.0, 1.0)
    val e by floatVar(0.0, 1.0)
    val f by floatVar(0.0, 1.0)
}
internal fun suiteProblem(name: String): RefinementProblem {
    val model = when (name) {
        "multimodal3" -> ThreeDimensions()
        "ellipsoid6" -> SixDimensions()
        else -> TwoDimensions()
    }
    return RefinementProblem(name, model) { point ->
        val x = point.sample.reals
        if (name == "branin2") {
            val a = 15 * x[0] - 5
            val b = 15 * x[1]
            val residual = b - 5.1 / (4 * PI * PI) * a * a + 5 / PI * a - 6
            maxOf(0.0, (residual * residual + 10 * (1 - 1 / (8 * PI)) * cos(a) + 10 - 0.39788735772973816) / 100)
        } else {
            val weights = DoubleArray(
                x.size,
            ) { if (name == "ellipsoid6") 10.0.pow(2.0 * it / (x.size - 1)) else (it + 1.0) }
            x.indices.sumOf { i ->
                val delta = x[i] - (0.25 + 0.08 * i)
                weights[i] * (delta * delta + if (name == "multimodal3") 0.02 * (1 - cos(20 * PI * delta)) else 0.0)
            } / weights.sum()
        }
    }
}
internal fun uniformGpWitnesses(dimensions: Int): WitnessSearch = WitnessSearch { request, random ->
    val keys = request.exclude.map { it.assignmentKey() }.toMutableSet()
    val points = mutableListOf<com.eignex.klause.solver.Sample>()
    repeat(request.maxAttempts) {
        if (points.size < request.count) {
            val sample = continuousPoint(DoubleArray(dimensions) { random.nextDouble() }).sample
            if (keys.add(sample.assignmentKey())) points += sample
        }
    }
    WitnessBatch(
        points,
        if (points.size == request.count) WitnessTermination.CountReached else WitnessTermination.BudgetExhausted,
    )
}

/** Paired acquisition/search ablations across objectives, dimensions, noise, and batch size. */
fun main(args: Array<String>) {
    val development = args.firstOrNull() == "development"
    val seeds = args.getOrNull(1)?.takeIf { it.isNotBlank() }?.split(',')?.map(String::toInt)
        ?: if (development) listOf(951, 952) else (1701..1706).toList()
    val output = File(
        "build/reports/gp-refinement/${if (development) "development" else "validation"}",
    ).also { it.mkdirs() }
    File(
        output,
        "outcomes.csv",
    ).writeText(
        "problem,seed,sd,batch,policy,calls,best_regret,recommendation_regret,model_fits," +
            "suggestion_nanos,recommendation_nanos,acquisition_evaluations\n",
    )
    File(output, "traces.csv").writeText("problem,seed,sd,batch,policy,call,round,observed,truth,coordinates\n")
    for (name in listOf("quadratic2", "branin2", "multimodal3", "ellipsoid6")) {
        for (seed in seeds) {
            for (sd in listOf(0.0, 0.03)) {
                for (batch in listOf(1, 2, 4)) {
                    for (policy in listOf(
                        "random",
                        "sequential-local",
                        "nei-pool",
                        "lognei-pool",
                        "nei-local",
                        "lognei-local",
                    )) {
                        runSuite(name, seed, sd, batch, policy, output)
                    }
                }
            }
            println("GP refinement problem=$name seed=$seed")
        }
    }
    scoreFrozenReference()
}

internal fun runSuite(
    name: String,
    seed: Int,
    sd: Double,
    batch: Int,
    policy: String,
    output: File,
    sampling: JointNormalSampling = JointNormalSampling.Antithetic,
) {
    val label = if (sampling == JointNormalSampling.Antithetic) policy else "$policy-sobol"
    val problem = suiteProblem(name)
    val projection = SearchTransform(problem.space)
    val dimensions = projection.featureSize
    var round = 0
    val pool = WitnessSearch { request, _ ->
        val random = Random(seed.toLong() * 1000003 + round)
        val excluded = request.exclude.map { it.assignmentKey() }.toSet()
        val points = List(16) { continuousPoint(DoubleArray(dimensions) { random.nextDouble() }).sample }
            .filter { it.assignmentKey() !in excluded }.take(request.count)
        WitnessBatch(
            points,
            if (points.size == request.count) WitnessTermination.CountReached else WitnessTermination.BudgetExhausted,
        )
    }
    val local = LocalSearchPolicy(starts = 2)
    val joint = if (policy.endsWith("-local")) {
        JointLocalAndRandomSearch(
            projection,
            pool,
            candidateCount = 16,
            passes = 1,
            localSteps = 4,
            neighbors = 6,
            localPolicy = local,
        )
    } else {
        JointPoolSearch(projection, pool, candidateCount = 16, starts = 2, passes = 1)
    }
    val optimizer = BayesianOptimizer(
        projection, GaussianProcessSurrogate(),
        acquisitionOptimizer = LocalAndRandomSearch(
            projection,
            pool,
            candidateCount = 16,
            localSteps = 4,
            neighbors = 6,
            localPolicy = local,
        ),
        batchPolicy = if (policy == "random" ||
            policy == "sequential-local"
        ) {
            SequentialBatchPolicy()
        } else {
            JointBatchPolicy(
                if (policy.startsWith("log")) {
                    JointLogNoisyExpectedImprovement(sampling)
                } else {
                    JointNoisyExpectedImprovement(sampling)
                },
                joint,
            )
        },
        initialDesign = LatinHypercubeInitialDesign(
            problem.space,
            32,
            seed = seed,
            probability = if (policy == "random") 1.0 else 0.2,
            witnesses = uniformGpWitnesses(dimensions),
        ),
        outcome = OutcomeSpec(
            maximize = false,
            noise = if (sd > 0.0) ObservationNoise.Noisy else ObservationNoise.Deterministic,
            standardize = true,
        ),
        exploration = uniformGpWitnesses(dimensions), randomSeed = seed, repeatPolicy = RepeatPolicy.Never,
    )
    var best = Double.POSITIVE_INFINITY
    var suggestions = 0L
    var evaluations = 0
    var calls = 0
    val trace = StringBuilder()
    while (calls < 32) {
        round++
        val start = System.nanoTime()
        val trials = optimizer.suggest(batch)
        suggestions += System.nanoTime() - start
        check(trials.size == batch)
        evaluations += optimizer.lastSuggestion!!.acquisitionEvaluations
        val measured = trials.mapIndexed { slot, trial ->
            noisyMeasurement(
                problem,
                trial.sample,
                seed,
                calls + slot,
                sd,
            )
        }
        for (slot in (0 until batch).reversed()) {
            optimizer.complete(
                trials[slot].id,
                measured[slot],
                noiseVariance = sd * sd,
            )
        }
        for ((slot, trial) in trials.withIndex()) {
            calls++
            val truth = problem.truth(trial.sample)
            best = minOf(best, truth)
            trace.append(
                "$name,$seed,$sd,$batch,$label,$calls,$round,${measured[slot]},$truth," +
                    "${trial.sample.sample.reals.joinToString(
                        ";",
                    )}\n",
            )
        }
    }
    val recommendStart = System.nanoTime()
    val recommendation = optimizer.recommendOrThrow(
        LocalAndRandomSearch(
            projection,
            uniformGpWitnesses(dimensions),
            candidateCount = 128,
            maxAttempts = 256,
            localSteps = 4,
            neighbors = 8,
        ),
    )
    val recommendNanos = System.nanoTime() - recommendStart
    File(output, "traces.csv").appendText(trace.toString())
    File(
        output,
        "outcomes.csv",
    ).appendText(
        "$name,$seed,$sd,$batch,$label,$calls,$best,${problem.truth(
            recommendation,
        )},${optimizer.fitCount},$suggestions,$recommendNanos,$evaluations\n",
    )
}

private fun scoreFrozenReference() {
    val rows = File("benchmarks/gp-refinement/reference.csv").readLines().drop(1).map { it.split(',') }
    val output = File("build/reports/gp-refinement/frozen-scores.csv")
    output.writeText("kind,q,count,fixed_mae,high_sample_mae,zero_fraction\n")
    for (kind in listOf("qei", "qnei")) {
        for (q in listOf("1", "2")) {
            val selected = rows.filter { it[0].endsWith(kind) && it[1] == q }
            val fixed = MaeLossStat()
            val precise = MaeLossStat()
            val zeros = MeanStat()
            for (row in selected) {
                fixed.update(row[3].toDouble(), row[4].toDouble(), 0L)
                precise.update(row[3].toDouble(), row[5].toDouble(), 0L)
                zeros.update(if (row[3].toDouble() == 0.0) 1.0 else 0.0, 0L)
            }
            output.appendText(
                "$kind,$q,${selected.size},${fixed.read(0L).mean},${precise.read(0L).mean},${zeros.read(0L).mean}\n",
            )
        }
    }
}
