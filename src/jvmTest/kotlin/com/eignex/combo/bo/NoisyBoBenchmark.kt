package com.eignex.combo.bo

import com.eignex.combo.bo.model.BatchForestConfig
import com.eignex.combo.bo.model.BatchSplitStrategy
import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.shared.WitnessBatch
import com.eignex.combo.shared.WitnessSearch
import com.eignex.combo.shared.WitnessTermination
import com.eignex.combo.shared.assignmentKey
import com.eignex.kumulant.math.nextNormal
import com.eignex.kumulant.stat.score.MseLossStat
import com.eignex.kumulant.stat.summary.MeanStat
import java.io.File
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random

internal fun noisyWitnesses(): WitnessSearch = WitnessSearch { request, random ->
    val excluded = request.exclude.map { it.assignmentKey() }.toMutableSet()
    val points = mutableListOf<com.eignex.klause.solver.Sample>()
    repeat(request.maxAttempts) {
        if (points.size < request.count) {
            val sample = continuousPoint(doubleArrayOf(random.nextDouble(), random.nextDouble())).sample
            if (excluded.add(sample.assignmentKey())) points += sample
        }
    }
    WitnessBatch(
        points,
        if (points.size == request.count) WitnessTermination.CountReached else WitnessTermination.BudgetExhausted,
    )
}

internal fun noisySearch(problem: RefinementProblem, count: Int = 64): AcquisitionOptimizer = LocalAndRandomSearch(
    SearchTransform(problem.space),
    noisyWitnesses(),
    candidateCount = count,
    maxAttempts = count * 2,
    localSteps = 4,
    neighbors = 8,
    localPolicy = LocalSearchPolicy(starts = 4),
)

internal fun noisyMeasurement(
    problem: RefinementProblem,
    sample: BanditSample,
    seed: Int,
    key: Int,
    sd: Double,
): Double {
    var hash = seed.toLong() * -7046029254386353131L + key
    for (value in sample.sample.reals) hash = hash * 31 + value.toBits()
    return problem.truth(sample) + Random(hash).nextNormal(0.0, sd)
}

private fun noisyOptimizer(
    problem: RefinementProblem,
    seed: Int,
    budget: Int,
    model: String,
    measurementsPerConfiguration: Double = 1.0,
): BayesianOptimizer = BayesianOptimizer(
    SearchTransform(problem.space),
    if (model == "gp") {
        GaussianProcessSurrogate()
    } else {
        ForestSurrogate(
            BatchForestConfig(splitStrategy = BatchSplitStrategy.Random, mtry = 2),
        )
    },
    randomSeed = seed,
    outcome = OutcomeSpec(maximize = false, noise = ObservationNoise.Noisy, standardize = true),
    acquisitionOptimizer = noisySearch(problem),
    exploration = noisyWitnesses(),
    initialDesign = LatinHypercubeInitialDesign(
        problem.space,
        budget,
        seed = seed,
        probability = 0.2,
        measurementsPerConfiguration = measurementsPerConfiguration,
    ),
    repeatPolicy = RepeatPolicy.Never,
)

internal fun diagnoseNoisy(
    optimizer: BayesianOptimizer,
    problem: RefinementProblem,
    seed: Int,
    sd: Double,
    group: String,
    output: File,
) {
    val small = optimizer.recommendOrThrow()
    val large = optimizer.recommendOrThrow(noisySearch(problem, 1024))
    val model = optimizer.fittedModel()
    val random = Random(seed xor 73471)
    val points = List(4096) { continuousPoint(doubleArrayOf(random.nextDouble(), random.nextDouble())) } +
        optimizer.store.snapshot().observations.map { it.sample } + listOf(small, large)
    val moments = model.predict(featureRows(points.map { optimizer.projection.encode(it) }, 2))
    val best = moments.means.toDoubleArray().indices.minBy { moments.means[it] }
    val mse = MseLossStat()
    val latent = MeanStat()
    val observed = MeanStat()
    val topMse = MseLossStat()
    val topCoverage = MeanStat()
    val threshold = points.take(4096).map(problem.truth).sorted()[409]
    for (i in 0 until 4096) {
        val truth = problem.truth(points[i])
        mse.update(moments.means[i], truth, 0L)
        if (truth <= threshold) {
            topMse.update(moments.means[i], truth, 0L)
            topCoverage.update(if (abs(moments.means[i] - truth) <= 2 * sqrt(moments.variances[i])) 1.0 else 0.0, 0L)
        }
        latent.update(if (abs(moments.means[i] - truth) <= 2 * sqrt(moments.variances[i])) 1.0 else 0.0, 0L)
        val measurement = noisyMeasurement(problem, points[i], seed, 90001 + i, sd)
        observed.update(
            if (abs(moments.means[i] - measurement) <= 2 * sqrt(moments.variances[i] + sd * sd)) 1.0 else 0.0,
            0L,
        )
    }
    val start = points.size - 2
    File(output, "diagnostics.csv").appendText(
        "$group,$seed,$sd,${optimizer.store.revision},${sqrt(mse.read(0L).mean)},${latent.read(0L).mean}," +
            "${observed.read(0L).mean},${moments.means[start] - moments.means[best]}," +
            "${moments.means[start + 1] - moments.means[best]},${problem.truth(small) - 1}," +
            "${problem.truth(
                large,
            ) - 1},${problem.truth(points[best]) - 1},${sqrt(topMse.read(0L).mean)},${topCoverage.read(0L).mean}\n",
    )
}

private fun runOrdinary(seed: Int, budget: Int, sd: Double, model: String, output: File) {
    val problem = refinementProblem("noisy")
    val optimizer = noisyOptimizer(problem, seed, budget, model)
    var best = Double.POSITIVE_INFINITY
    val trace = mutableListOf<String>()
    repeat(budget) { step ->
        val trial = optimizer.suggest(1)[0]
        val value = noisyMeasurement(problem, trial.sample, seed, step, sd)
        optimizer.complete(trial.id, value, noiseVariance = sd * sd)
        best = minOf(best, problem.truth(trial.sample))
        trace += "$seed,$budget,$sd,$model,${step + 1},$value,${problem.truth(
            trial.sample,
        )},${trial.sample.sample.reals.joinToString(",")}"
    }
    diagnoseNoisy(optimizer, problem, seed, sd, model, output)
    val small = optimizer.recommendOrThrow()
    val large = optimizer.recommendOrThrow(noisySearch(problem, 1024))
    File(output, "outcomes.csv").appendText(
        "$seed,$budget,$sd,$model,$budget,${best - 1},${problem.truth(small) - 1},${problem.truth(large) - 1}\n",
    )
    File(output, "trace-$seed-$budget-$sd-$model.csv").writeText(
        "seed,budget,sd,algorithm,call,observed,truth,x,y\n" + trace.joinToString("\n", postfix = "\n"),
    )
}

private fun runConfirmation(
    seed: Int,
    budget: Int,
    sd: Double,
    policy: String,
    output: File,
    measurementAware: Boolean,
) {
    val problem = refinementProblem("noisy")
    val optimizer = noisyOptimizer(
        problem,
        seed,
        if (measurementAware && policy != "race") budget - 9 else budget,
        "gp",
        measurementsPerConfiguration = if (measurementAware) 3.0 else 1.0,
    )
    val race = RacingOptimizer(
        optimizer,
        (1..4).map { EvaluationKey(seed = it) },
        RacingPolicy(
            minMeasurements = 1,
            initialIncumbentMeasurements = 1,
            rejectionStandardErrors = 2.0,
            randomSeed = seed,
        ),
        evaluationBudget = budget,
    )
    var calls = 0
    var best = Double.POSITIVE_INFINITY
    val trace = mutableListOf<String>()
    fun evaluate() {
        val task = race.ask(1).tasks.single()
        val value = noisyMeasurement(problem, task.sample, seed, task.key.seed, sd)
        race.complete(task.id, value, noiseVariance = sd * sd)
        calls++
        best = minOf(best, problem.truth(task.sample))
        trace += "$seed,$budget,$sd,$policy,$calls,$value,${problem.truth(
            task.sample,
        )},${task.sample.sample.reals.joinToString(",")}"
    }
    repeat(if (policy == "race") budget else budget - 9) { evaluate() }
    if (policy != "race") {
        val finalists = race.configurations.sortedBy { optimizer.predict(it.sample) }.take(3).map { it.id }
        val keys = (1001..1003).map { EvaluationKey(seed = it) }
        if (policy == "model-confirm") {
            race.beginConfirmation(
                keys,
                finalists,
            )
        } else {
            race.beginConfirmation(keys, candidates = 3)
        }
        val count = race.exportData().confirmation!!.configurationIds.size * keys.size
        repeat(count) { evaluate() }
    }
    val recommendation = race.incumbent!!.sample
    File(output, "outcomes.csv").appendText(
        "$seed,$budget,$sd,$policy,$calls,${best - 1},${problem.truth(
            recommendation,
        ) - 1},${problem.truth(recommendation) - 1}\n",
    )
    File(output, "trace-$seed-$budget-$sd-$policy.csv").writeText(
        "seed,budget,sd,algorithm,call,observed,truth,x,y\n" + trace.joinToString("\n", postfix = "\n"),
    )
}

/** Paired noisy recommendation diagnostics; truth is used for evaluation only. */
fun main(args: Array<String>) {
    val seeds = args.firstOrNull()?.takeIf { it.isNotBlank() }?.split(',')?.map(String::toInt) ?: (1301..1312).toList()
    val measurementAware = args.getOrNull(1) != "legacy"
    val output = File("build/reports/noisy-bo" + if (measurementAware) "" else "-legacy").also { it.mkdirs() }
    File(
        output,
        "outcomes.csv",
    ).writeText("seed,budget,sd,algorithm,calls,best_regret,recommendation_regret,large_regret\n")
    File(output, "diagnostics.csv").writeText(
        "algorithm,seed,sd,history,rmse,latent_coverage,observation_coverage,small_gap,large_gap," +
            "small_regret,large_regret,pool_regret,top_rmse,top_latent_coverage\n",
    )
    for (seed in seeds) {
        for (budget in listOf(32, 64)) {
            for (sd in listOf(0.03, 0.1)) {
                for (model in listOf("gp", "forest")) runOrdinary(seed, budget, sd, model, output)
                for (policy in listOf(
                    "race",
                    "observed-confirm",
                    "model-confirm",
                )) {
                    runConfirmation(seed, budget, sd, policy, output, measurementAware)
                }
                println("noisy seed=$seed budget=$budget sd=$sd")
            }
        }
    }
}
