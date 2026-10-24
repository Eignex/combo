package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.decisions.SubSpace
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.combo.shared.WitnessBatch
import com.eignex.combo.shared.WitnessSearch
import com.eignex.combo.shared.WitnessTermination
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.Sample
import com.eignex.koblas.DenseVector
import com.eignex.kumulant.stat.regression.tree.BatchForestConfig
import com.eignex.kumulant.stat.regression.tree.BatchSplitStrategy
import com.eignex.kumulant.stat.score.MseLossStat
import com.eignex.kumulant.stat.summary.MeanStat
import com.eignex.kumulant.stat.summary.estimateMean
import java.io.File
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

private class RefinementContinuous : DecisionSpace() {
    val x by floatVar(0.0, 1.0)
}
private class RefinementNoisy : DecisionSpace() {
    val x by floatVar(0.0, 1.0)
    val y by floatVar(0.0, 1.0)
}
private class RefinementChild : SubSpace() {
    val size by intVar(1, 6)
}
private class RefinementConditional : DecisionSpace() {
    val x by floatVar(0.0, 1.0)
    val mode by nominal("slow", "fast")
    val child by optionalDecisionSpace(::RefinementChild)
}
private class RefinementProblem(val name: String, val model: DecisionSpace, val truth: (BanditSample) -> Double) {
    val space = model.compileSpace()
}
private fun sq(x: Double) = x * x
private fun refinementProblem(name: String): RefinementProblem {
    if (name == "conditional") {
        val model = RefinementConditional()
        val space = model.compileSpace()
        return RefinementProblem(name, model) {
            1.0 + sq(it.float(model.x, space) - 0.62) +
                (
                    if (space.compiled.decode(space.gateOf(model.child)!!, it.sample)) {
                        0.02 * sq(space.compiled.decode(model.child.size, it.sample) - 4.0)
                    } else {
                        0.3
                    }
                    ) + if (space.compiled.decode(model.mode, it.sample) == "fast") 0.0 else 0.1
        }
    }
    if (name == "noisy") {
        val model = RefinementNoisy()
        val space = model.compileSpace()
        return RefinementProblem(name, model) {
            1.0 + sq(it.float(model.x, space) - 0.31) + 3.0 * sq(it.float(model.y, space) - 0.72)
        }
    }
    val model = RefinementContinuous()
    val space = model.compileSpace()
    return RefinementProblem(name, model) {
        val x = it.float(model.x, space)
        1.0 + if (name == "logarithmic") sq((x - 0.5) * 2.0) else sq(x - 0.73)
    }
}
private fun measured(problem: RefinementProblem, point: BanditSample, key: Int): Double {
    val source = point.sample.reals
    val hash = sin(key * 12.9898 + source[0] * 78.233 + source.getOrElse(1) { 0.0 } * 39.425) * 43758.5453
    return problem.truth(point) + if (problem.name == "noisy") 0.05 * (2 * (hash - floor(hash)) - 1) else 0.0
}
private fun continuousPoint(values: DoubleArray): BanditSample = BanditSample(
    Sample(
        booleanArrayOf(),
        longArrayOf(),
        values,
        exactReals = values.map { BigFraction.ofDouble(it)!! },
    ),
)

private fun frozen(seed: Int, name: String, size: Int, output: File) {
    val problem = refinementProblem(name)
    val projection = SearchTransform(problem.space)
    val random = Random(seed)
    val training = List(size) { continuousPoint(DoubleArray(projection.featureSize) { random.nextDouble() }) }
    val candidates = List(10000) { continuousPoint(DoubleArray(projection.featureSize) { random.nextDouble() }) }
    val observed = training.mapIndexed { index, point -> measured(problem, point, index + 100) }
    val rows = (training + candidates).map { projection.encode(it) }
    val features = featureRows(rows.take(size), projection.featureSize)
    val queries = featureRows(rows.drop(size), projection.featureSize)
    val dataset = File(output, "frozen-$name-$seed-$size.csv")
    dataset.printWriter().use { writer ->
        writer.println("role,response,truth," + (0 until projection.featureSize).joinToString(",") { "x$it" })
        for ((index, sample) in (training + candidates).withIndex()) {
            writer.println(
                "${if (index < size) "train" else "query"},${if (index < size) observed[index] else ""}," +
                    "${problem.truth(sample)}," + rows[index].toDoubleArray().joinToString(","),
            )
        }
    }
    val config = BatchForestConfig(splitStrategy = BatchSplitStrategy.Random, mtry = projection.featureSize)
    for (uncertainty in ForestUncertainty.entries) {
        val data = TrainingData(
            features,
            DenseVector.of(observed.toDoubleArray()),
            DenseVector.of(DoubleArray(size) { 1.0 }),
            if (name == "noisy") DenseVector.of(DoubleArray(size) { 0.0025 / 3.0 }) else null,
        )
        val fitted = ForestSurrogate(config, uncertainty).fit(data, Random(seed))
        val moments = fitted.predict(queries)
        val loss = MseLossStat()
        val coverage = MeanStat()
        val width = MeanStat()
        val topLoss = MseLossStat()
        val topCoverage = MeanStat()
        val topThreshold = candidates.map(problem.truth).sorted()[candidates.size / 10]
        for (i in candidates.indices) {
            val truth = problem.truth(candidates[i])
            val deviation = sqrt(moments.variances[i])
            loss.update(moments.means[i], truth, 0L, 1.0)
            coverage.update(if (abs(moments.means[i] - truth) <= 2.0 * deviation) 1.0 else 0.0, 0L, 1.0)
            width.update(deviation, 0L, 1.0)
            if (truth <= topThreshold) {
                topLoss.update(moments.means[i], truth, 0L, 1.0)
                topCoverage.update(if (abs(moments.means[i] - truth) <= 2.0 * deviation) 1.0 else 0.0, 0L, 1.0)
            }
        }
        File(output, "models.csv").appendText(
            "$name,$seed,$size,$uncertainty,${sqrt(loss.read(0L).mean)}," +
                "${coverage.read(
                    0L,
                ).mean},${width.read(0L).mean},${sqrt(topLoss.read(0L).mean)},${topCoverage.read(0L).mean}\n",
        )
        val baseline = if (name == "noisy") fitted.predict(features).means.toDoubleArray().min() else observed.min()
        val acquisition = ExpectedImprovement().create(
            AcquisitionContext(fitted, baseline, OutcomeSpec(maximize = false)),
            Random(seed),
        )
        val oracle = acquisition.evaluate(queries).toDoubleArray().max()
        for (count in listOf(64, 256, 10000)) {
            val search = LocalAndRandomSearch(
                projection,
                WitnessSearch { request, _ ->
                    WitnessBatch(candidates.take(request.count).map { it.sample }, WitnessTermination.CountReached)
                },
                candidateCount = count,
                maxAttempts = 0,
                localSteps = 4,
                neighbors = 8,
                localPolicy = LocalSearchPolicy(starts = 4),
            )
            val selected = search.maximize(
                acquisition,
                AcquisitionRequest(
                    Context.Empty,
                    training.map { it.sample },
                    training.map { it.sample },
                ),
                Random(seed),
            )
            val score = acquisition.evaluate(
                featureRows(listOf(projection.encode(selected.sample!!)), projection.featureSize),
            )[0]
            File(output, "search.csv").appendText(
                "$name,$seed,$size,$uncertainty,$count,${selected.evaluations}," +
                    "$oracle,$score,${problem.truth(selected.sample)}\n",
            )
        }
    }
    println("frozen $name seed=$seed n=$size")
}

private fun optimize(seed: Int, name: String, budget: Int, algorithm: String, output: File) {
    val problem = refinementProblem(name)
    val projection = SearchTransform(problem.space)
    val base = RandomForestOptimizer(
        projection,
        config = ForestOptimizerConfig(
            forest = BatchForestConfig(splitStrategy = BatchSplitStrategy.Random, mtry = projection.featureSize),
            candidateCount = 64, maxWitnessAttempts = 128, localSteps = 4, neighbors = 8,
            localSearch = LocalSearchPolicy(starts = 4), randomExploration = 0.2,
            repeatPolicy = RepeatPolicy.Never,
            uncertainty = if (algorithm == "leaf") ForestUncertainty.LeafMean else ForestUncertainty.Disagreement,
        ),
        randomSeed = seed,
        maximize = false,
        outcome = OutcomeSpec(
            maximize = false,
            noise = if (name == "noisy") ObservationNoise.Noisy else ObservationNoise.Deterministic,
            responseScale = ResponseScale.Log,
        ),
        initialDesign = LatinHypercubeInitialDesign(
            problem.space,
            budget,
            seed = seed,
            measurementsPerConfiguration = if (algorithm in listOf(
                    "budgeted-race",
                    "guarded-race",
                    "confirmed-race",
                )
            ) {
                3.0
            } else {
                1.0
            },
        ),
    )
    val runner: TrialOptimizer = if (algorithm == "random") {
        StudyRunner(
            Study(problem.space),
            RandomSuggestionPolicy(problem.space, HybridDescentSession.klause(problem.space), seed),
        )
    } else {
        base
    }
    val race = if (algorithm.contains("race") || algorithm == "confirmation") {
        RacingOptimizer(
            base,
            (if (algorithm == "confirmation") 0..0 else 0..2).map { EvaluationKey(seed = it) },
            policy = if (algorithm in listOf(
                    "legacy-race",
                    "budgeted-race",
                    "confirmation",
                )
            ) {
                RacingPolicy(randomSeed = seed)
            } else {
                RacingPolicy(
                    minMeasurements = 2,
                    initialIncumbentMeasurements = 2,
                    rejectionStandardErrors = 2.0,
                    randomSeed = seed,
                )
            },
            evaluationBudget = budget,
        )
    } else {
        null
    }
    val confirmationMeasurements = maxOf(1, budget / 5 / 3)
    val confirmationBudget = confirmationMeasurements * 3
    val confirm = algorithm == "confirmed-race" || algorithm == "confirmation"
    val lines = mutableListOf(
        "problem,algorithm,seed,budget,step,best_regret,recommendation_regret,measurements,distinct_configurations",
    )
    var best = Double.POSITIVE_INFINITY
    val points = linkedMapOf<BanditSample, MutableList<Double>>()
    repeat(budget) { index ->
        if (confirm && index == budget - confirmationBudget) {
            race!!.beginConfirmation(List(confirmationMeasurements) { EvaluationKey(seed = 1000 + it) })
        }
        val task = race?.ask(1)?.tasks?.firstOrNull()
        val trial = if (race == null) runner.suggest(1).trials.firstOrNull() else null
        val sample = task?.sample ?: trial?.sample ?: error("$algorithm stopped at $index of $budget")
        val value = measured(problem, sample, task?.key?.seed ?: 0)
        if (task != null) {
            race.complete(task.id, value, noiseVariance = 0.0025 / 3.0)
        } else {
            runner.complete(trial!!.id, value, noiseVariance = if (name == "noisy") 0.0025 / 3.0 else null)
        }
        best = minOf(best, problem.truth(sample))
        points.getOrPut(sample) { mutableListOf() }.add(value)
        val recommended = if (race != null) {
            race.incumbent?.sample
        } else {
            points.minBy { estimateMean(DenseVector.of(it.value.toDoubleArray())).mean }.key
        }
        val regret = recommended?.let { (problem.truth(it) - 1.0).toString() } ?: ""
        lines +=
            "$name,$algorithm,$seed,$budget,${index + 1},${best - 1.0},$regret," +
            "${race?.incumbent?.measurements ?: 1},${points.size}"
    }
    File(output, "$algorithm-$name-$seed-$budget.csv").writeText(lines.joinToString("\n", postfix = "\n"))
    println("optimize $name $algorithm seed=$seed budget=$budget regret=${best - 1.0}")
}

/** Optional model and allocation diagnostics, kept outside the unit test gate. */
fun main(args: Array<String>) {
    val mode = args.firstOrNull() ?: "all"
    val output = File("build/reports/bo-refinement").also { it.mkdirs() }
    val seeds = args.getOrNull(1)?.takeIf { it.isNotBlank() }?.split(',')?.map(String::toInt)
    if (mode == "frozen" || mode == "all") {
        File(
            output,
            "models.csv",
        ).writeText("problem,seed,history,uncertainty,rmse,coverage_two_sd,mean_sd,top_rmse,top_coverage_two_sd\n")
        File(
            output,
            "search.csv",
        ).writeText(
            "problem,seed,history,uncertainty,candidates,evaluations," +
                "reference_acquisition,selected_acquisition,selected_truth\n",
        )
        for (seed in seeds ?: (201..220).toList()) {
            for (name in listOf("quadratic", "noisy")) {
                for (n in listOf(32, 128)) frozen(seed, name, n, output)
            }
        }
    }
    if (mode == "optimize" || mode == "all") {
        for (seed in seeds ?: (101..112).toList()) {
            for (name in listOf("quadratic", "conditional", "noisy")) {
                for (budget in listOf(32, 128)) {
                    for (algorithm in listOf("disagreement", "leaf", "random")) {
                        optimize(seed, name, budget, algorithm, output)
                    }
                }
            }
        }
    }
    if (mode == "race" || mode == "all") {
        for (seed in seeds ?: (101..112).toList()) {
            for (budget in listOf(32, 128)) {
                for (algorithm in listOf(
                    "legacy-race",
                    "budgeted-race",
                    "guarded-race",
                    "confirmed-race",
                    "confirmation",
                )) {
                    optimize(seed, "noisy", budget, algorithm, output)
                }
            }
        }
    }
}
