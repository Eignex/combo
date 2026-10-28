package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.decisions.Scale
import com.eignex.combo.shared.HybridDescentSession
import java.io.File
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.sin

private class QuadraticComparison : DecisionSpace() {
    val x by floatVar(0.0, 1.0)
}
private class LogComparison : DecisionSpace() {
    val x by floatVar(1e-6, 1.0, scale = Scale.Log)
}
private class MixedComparison : DecisionSpace() {
    val x by floatVar(0.0, 1.0)
    val count by intVar(1, 6)
    val mode by nominal("slow", "fast")
}
private class NoisyComparison : DecisionSpace() {
    val x by floatVar(0.0, 1.0)
    val y by floatVar(0.0, 1.0)
}
private class ComparisonProblem(
    val name: String,
    val space: CompiledDecisionSpace,
    val coordinates: (BanditSample) -> List<String>,
    val reference: (BanditSample) -> Double,
)
private fun square(value: Double) = value * value
private fun problem(name: String): ComparisonProblem = when (name) {
    "quadratic" -> {
        val model = QuadraticComparison()
        val space = model.compileSpace()
        ComparisonProblem(
            name,
            space,
            { listOf(it.float(model.x, space).toString(), "", "", "") },
            { 1.0 + square(it.float(model.x, space) - 0.73) },
        )
    }

    "logarithmic" -> {
        val model = LogComparison()
        val space = model.compileSpace()
        ComparisonProblem(
            name,
            space,
            { listOf(it.float(model.x, space).toString(), "", "", "") },
            { 1.0 + square(ln(it.float(model.x, space) / 1e-3) / ln(1000.0)) },
        )
    }

    "mixed" -> {
        val model = MixedComparison()
        val space = model.compileSpace()
        ComparisonProblem(
            name,
            space,
            {
                listOf(
                    it.float(model.x, space).toString(),
                    "",
                    space.compiled.decode(model.count, it.sample).toString(),
                    space.compiled.decode(model.mode, it.sample),
                )
            },
            {
                1.0 + square(it.float(model.x, space) - 0.65) +
                    square((space.compiled.decode(model.count, it.sample) - 4).toDouble() / 6.0) +
                    if (space.compiled.decode(model.mode, it.sample) == "fast") 0.0 else 0.15
            },
        )
    }

    else -> {
        val model = NoisyComparison()
        val space = model.compileSpace()
        ComparisonProblem(
            name,
            space,
            { listOf(it.float(model.x, space).toString(), it.float(model.y, space).toString(), "", "") },
            { 1.0 + square(it.float(model.x, space) - 0.31) + 3.0 * square(it.float(model.y, space) - 0.72) },
        )
    }
}
private fun measurementNoise(coordinates: List<String>, seed: Int): Double {
    val value = sin(
        seed * 12.9898 + coordinates[0].toDouble() * 78.233 + coordinates[1].toDouble() * 39.425,
    ) * 43758.5453
    return 0.05 * (2.0 * (value - floor(value)) - 1.0)
}
private fun measurementSeeds(output: File, seed: Int): List<Int> {
    val lines = File(output, "smac-noisy-$seed.csv").readLines()
    val column = lines.first().split(',').indexOf("eval_seed")
    return lines.drop(1).map { it.split(',')[column].toInt() }.distinct()
}
private fun compare(name: String, seed: Int, algorithm: String, output: File) {
    val problem = problem(name)
    val noisy = name == "noisy"
    val seeds = if (noisy) measurementSeeds(output, seed) else listOf(0)
    val optimizer = RandomForestOptimizer.smacProfile(
        problem.space,
        32,
        seed,
        noise = if (noisy) ObservationNoise.Noisy else ObservationNoise.Deterministic,
        repeatPolicy = RepeatPolicy.Never,
    )
    val racing = if (algorithm == "combo-racing") {
        RacingOptimizer(
            optimizer,
            seeds.map { EvaluationKey(seed = it) },
            policy = RacingPolicy(randomSeed = seed),
            evaluationBudget = 32,
        )
    } else {
        null
    }
    val runner: TrialOptimizer = if (algorithm == "random") {
        StudyRunner(
            Study(problem.space),
            RandomSuggestionPolicy(problem.space, HybridDescentSession.klause(problem.space), seed),
        )
    } else {
        optimizer
    }
    val rows = mutableListOf(
        "problem,algorithm,seed,evaluations,cumulative_cost,observed_response,best_reference," +
            "simple_regret,x,y,count,mode,eval_seed,recommendation_regret,duration_seconds",
    )
    var best = Double.POSITIVE_INFINITY
    val observed = linkedMapOf<List<String>, Pair<Double, Int>>()
    val references = mutableMapOf<List<String>, Double>()
    repeat(32) { step ->
        val started = System.nanoTime()
        val task = racing?.ask(1)?.tasks?.firstOrNull()
        val trial = if (racing == null) runner.suggest(1).trials.firstOrNull() else null
        val sample = task?.sample ?: trial?.sample ?: error("comparison stopped before its evaluation budget")
        val coordinates = problem.coordinates(sample)
        val measurementSeed = task?.key?.seed ?: seeds.first()
        val truth = problem.reference(sample)
        val measured = truth + if (noisy) measurementNoise(coordinates, measurementSeed) else 0.0
        if (task != null) racing.complete(task.id, measured) else runner.complete(trial!!.id, measured)
        best = minOf(best, truth)
        val previous = observed[coordinates] ?: (0.0 to 0)
        observed[coordinates] = previous.first + measured to previous.second + 1
        references[coordinates] = truth
        val recommended = if (racing == null) {
            references.getValue(observed.minBy { it.value.first / it.value.second }.key)
        } else {
            racing.incumbent?.sample?.let(problem.reference)
        }
        val recommendationRegret = recommended?.let { maxOf(0.0, it - 1.0).toString() } ?: ""
        rows += "$name,$algorithm,$seed,${step + 1},${step + 1},$measured,$best,${maxOf(0.0, best - 1.0)}," +
            coordinates.joinToString(
                ",",
            ) + ",$measurementSeed,$recommendationRegret,${(System.nanoTime() - started) / 1e9}"
    }
    File(output, "$algorithm-$name-$seed.csv").writeText(rows.joinToString("\n", postfix = "\n"))
    println("$algorithm $name seed=$seed regret=${best - 1.0}")
}

/** Optional external benchmark entry point, separate from the normal JVM test gate. */
fun main() {
    val output = File("build/reports/smac-comparison").also { it.mkdirs() }
    for (name in listOf("quadratic", "logarithmic", "mixed", "noisy")) {
        for (seed in listOf(3, 11, 19, 29, 43, 61)) {
            for (algorithm in listOf(
                "combo-profile",
                "random",
            ) + if (name == "noisy") listOf("combo-racing") else emptyList()) {
                compare(name, seed, algorithm, output)
                if (seed == 3) {
                    val file = File(output, "$algorithm-$name-$seed.csv")
                    val expected = file.readLines().map { it.substringBeforeLast(',') }
                    compare(name, seed, algorithm, output)
                    check(expected == file.readLines().map { it.substringBeforeLast(',') }) {
                        "$algorithm $name did not replay exactly"
                    }
                }
            }
        }
    }
}
