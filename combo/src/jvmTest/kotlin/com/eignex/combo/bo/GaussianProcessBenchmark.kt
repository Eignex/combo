package com.eignex.combo.bo

import com.eignex.combo.bo.model.BatchForestConfig
import com.eignex.combo.bo.model.BatchSplitStrategy
import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.combo.shared.WitnessBatch
import com.eignex.combo.shared.WitnessSearch
import com.eignex.combo.shared.WitnessTermination
import com.eignex.combo.shared.WitnessValidator
import com.eignex.combo.shared.assignmentKey
import com.eignex.koblas.DenseVector
import com.eignex.kumulant.stat.score.MseLossStat
import com.eignex.kumulant.stat.summary.MeanStat
import java.io.File
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

private val gpProblems = listOf("quadratic", "multimodal", "noisy", "logarithmic")
private fun gpOutcome(name: String) = OutcomeSpec(
    maximize = false,
    standardize = true,
    noise = if (name == "noisy") ObservationNoise.Noisy else ObservationNoise.Deterministic,
)
private fun gpNoise(name: String) = if (name == "noisy") 0.0025 / 3.0 else 0.0
private fun gpSurrogate(name: String, dimensions: Int): Surrogate = if (name == "gp") {
    GaussianProcessSurrogate()
} else {
    ForestSurrogate(BatchForestConfig(splitStrategy = BatchSplitStrategy.Random, mtry = dimensions))
}
private fun gpPoint(problem: RefinementProblem, random: Random): BanditSample {
    val dimensions = SearchTransform(problem.space).featureSize
    val values = DoubleArray(dimensions) { random.nextDouble() }
    if (problem.name == "logarithmic") values[0] = exp(ln(1e-6) + values[0] * ln(1e6))
    return continuousPoint(values)
}

private fun gpFrozen(name: String, seed: Int, count: Int, output: File) {
    val problem = refinementProblem(name)
    val projection = SearchTransform(problem.space)
    val random = Random(seed)
    val training = List(count) { gpPoint(problem, random) }
    val candidates = List(4096) { gpPoint(problem, random) }
    val observations = training.mapIndexed { index, point ->
        Observation(point, Context.Empty, measured(problem, point, index + 100), noiseVariance = gpNoise(name))
    }
    val response = ResponseTransform.from(observations, gpOutcome(name))
    val features = featureRows(training.map { projection.encode(it) }, projection.featureSize)
    val queries = featureRows(candidates.map { projection.encode(it) }, projection.featureSize)
    val data = TrainingData(
        features,
        DenseVector.of(DoubleArray(count) { response.encode(observations[it].reward) }),
        DenseVector.of(DoubleArray(count) { 1.0 }),
        DenseVector.of(DoubleArray(count) { response.encodeNoise(observations[it].reward, gpNoise(name)) }),
    )
    val top = candidates.map(problem.truth).sorted()[candidates.size / 10]
    for (algorithm in listOf("gp", "forest")) {
        val raw = gpSurrogate(algorithm, projection.featureSize).fit(data, Random(seed))
        val fitted = response.model(raw)
        val moments = fitted.predict(queries)
        val loss = MseLossStat()
        val topLoss = MseLossStat()
        val coverage = MeanStat()
        val width = MeanStat()
        for (i in candidates.indices) {
            val truth = problem.truth(candidates[i])
            loss.update(moments.means[i], truth, 0L, 1.0)
            if (truth <= top) topLoss.update(moments.means[i], truth, 0L, 1.0)
            val deviation = sqrt(moments.variances[i])
            coverage.update(if (abs(moments.means[i] - truth) <= 2.0 * deviation) 1.0 else 0.0, 0L, 1.0)
            width.update(deviation, 0L, 1.0)
        }
        val gp = raw as? FittedGaussianProcess
        File(output, "models.csv").appendText(
            "$name,$seed,$count,$algorithm,${sqrt(loss.read(0L).mean)},${sqrt(topLoss.read(0L).mean)}," +
                "${coverage.read(0L).mean},${width.read(0L).mean}," +
                "${gp?.parameters?.lengthScales?.joinToString(";") ?: ""}," +
                "${gp?.parameters?.signalVariance ?: ""},${gp?.numericalJitter ?: ""}\n",
        )
        val baseline = if (name == "noisy") {
            fitted.predict(
                features,
            ).means.toDoubleArray().min()
        } else {
            observations.minOf { it.reward }
        }
        val acquisition = ExpectedImprovement().create(
            AcquisitionContext(fitted, baseline, gpOutcome(name)),
            Random(seed),
        )
        val reference = acquisition.evaluate(queries).toDoubleArray().max()
        for (pool in listOf(64, 256)) {
            val search = LocalAndRandomSearch(
                projection,
                WitnessSearch { _, _ ->
                    WitnessBatch(
                        candidates.take(pool).map { it.sample },
                        WitnessTermination.CountReached,
                    )
                },
                candidateCount = pool,
                localSteps = 4,
                neighbors = 8,
                localPolicy = LocalSearchPolicy(starts = 4),
            )
            val selected = search.maximize(
                acquisition,
                AcquisitionRequest(Context.Empty, emptyList(), emptyList()),
                Random(seed),
            )
            val point = requireNotNull(selected.sample)
            val score = acquisition.evaluate(featureRows(listOf(projection.encode(point)), projection.featureSize))[0]
            File(
                output,
                "search.csv",
            ).appendText("$name,$seed,$count,$algorithm,$pool,$reference,$score,${problem.truth(point)}\n")
        }
    }
}

private fun gpOptimize(name: String, seed: Int, budget: Int, algorithm: String, output: File) {
    val problem = refinementProblem(name)
    val projection = SearchTransform(problem.space)
    val outcome = gpOutcome(name)
    val witnesses = HybridDescentSession.klause(problem.space, candidateBatchSize = 64, poolCapacity = 256)
    val validator = WitnessValidator(problem.space)
    val independent = WitnessSearch { request, random ->
        val keys = request.exclude.map { it.assignmentKey() }.toMutableSet()
        val points = mutableListOf<com.eignex.klause.solver.Sample>()
        for (attempt in 0 until request.maxAttempts) {
            if (points.size >= request.count) break
            val point = gpPoint(problem, random).sample
            if (validator.valid(point) && keys.add(point.assignmentKey())) points += point
        }
        WitnessBatch(
            points,
            if (points.size >= request.count) WitnessTermination.CountReached else WitnessTermination.BudgetExhausted,
        )
    }
    val optimizer = BayesianOptimizer(
        projection,
        gpSurrogate(algorithm, projection.featureSize),
        outcome = outcome,
        randomSeed = seed,
        initialDesign = LatinHypercubeInitialDesign(
            problem.space,
            budget,
            seed = seed,
            probability = if (algorithm == "random") 1.0 else 0.2,
        ),
        acquisitionOptimizer = LocalAndRandomSearch(
            projection,
            witnesses,
            candidateCount = 64,
            localSteps = 4,
            neighbors = 8,
            localPolicy = LocalSearchPolicy(starts = 4),
        ),
        repeatPolicy = RepeatPolicy.Never,
        exploration = independent,
    )
    val rows = mutableListOf("problem,seed,budget,algorithm,step,observed,truth,best_regret,coordinates")
    var best = Double.POSITIVE_INFINITY
    repeat(budget) { step ->
        val trial = optimizer.suggest(1)[0]
        val truth = problem.truth(trial.sample)
        val reward = measured(problem, trial.sample, seed * 1000 + step)
        optimizer.complete(trial.id, reward, noiseVariance = gpNoise(name))
        best = minOf(best, truth)
        rows += "$name,$seed,$budget,$algorithm,${step + 1},$reward,$truth,${best - 1.0}," +
            projection.encode(trial.sample).toDoubleArray().joinToString(";")
    }
    val recommendation = if (algorithm == "random") {
        requireNotNull(
            optimizer.bestObserved(),
        ).sample
    } else {
        optimizer.recommendOrThrow()
    }
    File(output, "outcomes.csv").appendText(
        "$name,$seed,$budget,$algorithm,${best - 1.0},${problem.truth(recommendation) - 1.0},${optimizer.fitCount}\n",
    )
    File(output, "trace-$name-$seed-$budget-$algorithm.csv").writeText(rows.joinToString("\n", postfix = "\n"))
    println("$name $algorithm seed=$seed budget=$budget regret=${best - 1.0}")
}

/** Optional paired model and optimization diagnostics; no stochastic ranking assertions enter unit tests. */
fun main(args: Array<String>) {
    val mode = args.firstOrNull() ?: "all"
    val seeds = args.getOrNull(1)?.takeIf(String::isNotBlank)?.split(',')?.map(String::toInt)
    val output = File("build/reports/gp-comparison").also { it.mkdirs() }
    if (mode in listOf("all", "frozen")) {
        File(
            output,
            "models.csv",
        ).writeText(
            "problem,seed,history,algorithm,rmse,top_rmse,coverage_two_sd,mean_sd," +
                "length_scales,signal_variance,jitter\n",
        )
        File(
            output,
            "search.csv",
        ).writeText("problem,seed,history,algorithm,pool,reference_acquisition,selected_acquisition,selected_truth\n")
        for (name in gpProblems) {
            for (seed in seeds ?: (401..408).toList()) {
                for (count in listOf(16, 64)) gpFrozen(name, seed, count, output)
            }
        }
    }
    if (mode in listOf("all", "optimize")) {
        File(output, "outcomes.csv").writeText("problem,seed,budget,algorithm,best_regret,recommendation_regret,fits\n")
        for (name in gpProblems) {
            for (seed in seeds ?: (301..308).toList()) {
                for (budget in listOf(32, 64)) {
                    for (algorithm in listOf("gp", "forest", "random")) {
                        gpOptimize(name, seed, budget, algorithm, output)
                    }
                }
            }
        }
    }
}
