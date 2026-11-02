package com.eignex.combo.bench

import com.eignex.combo.bo.BayesianOptimizer
import com.eignex.combo.bo.ForestSurrogate
import com.eignex.combo.bo.GaussianProcessSurrogate
import com.eignex.combo.bo.LatinHypercubeInitialDesign
import com.eignex.combo.bo.LocalAndRandomSearch
import com.eignex.combo.bo.LocalSearchPolicy
import com.eignex.combo.bo.ObservationNoise
import com.eignex.combo.bo.OutcomeSpec
import com.eignex.combo.bo.RepeatPolicy
import com.eignex.combo.bo.SearchTransform
import com.eignex.combo.bo.model.BatchForestConfig
import com.eignex.combo.bo.model.BatchSplitStrategy
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.lab.bench.BenchCase
import com.eignex.lab.bench.BenchCli
import com.eignex.lab.bench.BenchContext
import com.eignex.lab.bench.BenchDescriptor
import com.eignex.lab.bench.BenchMetric
import com.eignex.lab.bench.BenchRequest
import com.eignex.lab.bench.BenchResources
import com.eignex.lab.bench.BenchResult
import com.eignex.lab.bench.BenchSuite
import com.eignex.lab.bench.MetricDirection
import java.nio.file.Files
import kotlin.random.Random

fun main(args: Array<String>) {
    if (args.firstOrNull() == "analyze") com.eignex.combo.bench.diagnostics.AnalyzeDiagnostics.execute(args.drop(1).toTypedArray())
    else BenchCli.execute(ComboLabSuite, args)
}

internal object ComboLabSuite : BenchSuite {
    private val workloads = setOf("quadratic", "multimodal", "logarithmic", "noisy", "conditional")
    private val algorithms = setOf("gp", "forest", "random")
    private val options = setOf("workloads", "algorithms", "seeds", "repetitions", "budget", "smoke", "memory-mib", "timeout-seconds")

    override fun describe() = BenchDescriptor("combo-bench", "Paired synthetic Bayesian optimization with evaluation traces")

    override fun plan(request: BenchRequest): List<BenchCase> {
        if ("study" in request.parameters) return com.eignex.combo.bench.diagnostics.DiagnosticSuite.plan(request)
        val p = request.parameters
        require(p.keys.all { it in options }) { "Unknown Combo bench parameters: ${p.keys - options}" }
        val smoke = p["smoke"]?.toBooleanStrict() ?: false
        val selected = selection(p["workloads"] ?: if (smoke) "quadratic" else workloads.joinToString(","), workloads)
        val arms = selection(p["algorithms"] ?: "gp,forest,random", algorithms)
        val seeds = (p["seeds"] ?: if (smoke) "3" else "3,11,19,29,43,61").split(',').map {
            requireNotNull(it.toIntOrNull()) { "seeds must contain integers" }
        }
        require(seeds.distinct().size == seeds.size) { "Duplicate seed" }
        val repetitions = positive(p, "repetitions", 1)
        val budget = positive(p, "budget", if (smoke) 4 else 32)
        val memory = positive(p, "memory-mib", 2048)
        require(memory >= 2048) { "memory-mib must be at least 2048 for the 1 GiB JVM heap" }
        val resources = BenchResources(1, memory.toLong(), positive(p, "timeout-seconds", 600).toLong())
        return selected.flatMap { name -> seeds.flatMap { seed -> (1..repetitions).map { repetition ->
            BenchCase("$name/$seed/$repetition", mapOf(
                "workload" to name, "seed" to seed.toString(), "repetition" to repetition.toString(),
                "algorithms" to arms.joinToString(","), "budget" to budget.toString(),
            ), resources)
        } } }
    }

    private fun selection(value: String, allowed: Set<String>): List<String> = value.split(',').also { selected ->
        require(selected.distinct().size == selected.size && selected.all { it in allowed }) {
            "Unknown or duplicate selection: $value; expected $allowed"
        }
    }

    private fun positive(p: Map<String, String>, key: String, default: Int): Int =
        (p[key] ?: default.toString()).toIntOrNull()?.also { require(it > 0) { "$key must be positive" } }
            ?: throw IllegalArgumentException("$key must be a positive integer")

    override fun run(case: BenchCase, context: BenchContext): BenchResult {
        if ("study" in case.parameters) return com.eignex.combo.bench.diagnostics.DiagnosticSuite.run(case, context)
        val p = case.parameters
        require(p.keys == setOf("workload", "seed", "repetition", "algorithms", "budget")) { "Invalid frozen parameters" }
        val expected = plan(BenchRequest(mapOf(
            "workloads" to p.getValue("workload"), "seeds" to p.getValue("seed"),
            "repetitions" to "1", "algorithms" to p.getValue("algorithms"), "budget" to p.getValue("budget"),
            "memory-mib" to case.resources.memoryMiB.toString(), "timeout-seconds" to case.resources.timeoutSeconds.toString(),
        ))).single()
        val repetition = positive(p, "repetition", 1)
        require(case.id == "${p.getValue("workload")}/${p.getValue("seed")}/$repetition" &&
            case.resources == expected.resources && case.inputs.isEmpty() &&
            p == expected.parameters + ("repetition" to repetition.toString())) { "Case does not match its frozen workload" }
        val report = Files.createDirectory(context.reportPath())
        Files.writeString(report.resolve("runtime.txt"),
            "java=${System.getProperty("java.runtime.version")}\nos=${System.getProperty("os.name")}/${System.getProperty("os.arch")}\nparameters=$p\n")
        val metrics = mutableListOf<BenchMetric>()
        // Rotate arm order across repetitions; every arm remains inside this worker attempt.
        val arms = p.getValue("algorithms").split(',')
        val offset = (repetition - 1) % arms.size
        for (algorithm in arms.drop(offset) + arms.take(offset)) {
            val problem = workload(p.getValue("workload"))
            val projection = SearchTransform(problem.space)
            val seed = p.getValue("seed").toInt()
            val budget = p.getValue("budget").toInt()
            val noisy = p.getValue("workload") == "noisy"
            val noise = Random(seed xor 0x6e6f6973)
            val optimizer = BayesianOptimizer(
                projection,
                if (algorithm == "gp") GaussianProcessSurrogate() else ForestSurrogate(
                    BatchForestConfig(splitStrategy = BatchSplitStrategy.Random, mtry = projection.featureSize)),
                outcome = OutcomeSpec(maximize = false, standardize = true,
                    noise = if (noisy) ObservationNoise.Noisy else ObservationNoise.Deterministic),
                randomSeed = seed,
                initialDesign = LatinHypercubeInitialDesign(problem.space, budget, seed = seed,
                    probability = if (algorithm == "random") 1.0 else 0.2),
                acquisitionOptimizer = LocalAndRandomSearch(projection,
                    HybridDescentSession.klause(problem.space, candidateBatchSize = 64, poolCapacity = 256),
                    candidateCount = 64, localSteps = 4, neighbors = 8, localPolicy = LocalSearchPolicy(starts = 4)),
                repeatPolicy = RepeatPolicy.Never,
            )
            var best = Double.POSITIVE_INFINITY
            var acquisitionWork = 0L
            val start = System.nanoTime()
            Files.newBufferedWriter(report.resolve("$algorithm.csv")).use { writer ->
                writer.appendLine("step,cumulative_cost,observed,truth,best_regret,model_fits,acquisition_evaluations,coordinates")
                repeat(budget) { step ->
                    val batch = optimizer.suggest(1)
                    check(batch.size == 1) { "Incomplete evaluation budget for $algorithm at step $step" }
                    val trial = batch[0]
                    val truth = problem.truth(trial.sample)
                    val observed = truth + if (noisy) noise.nextDouble(-0.05, 0.05) else 0.0
                    check(truth.isFinite() && observed.isFinite() && truth >= 1.0) { "Invalid synthetic evaluation" }
                    optimizer.complete(trial.id, observed, noiseVariance = if (noisy) 0.0025 / 3.0 else 0.0)
                    best = minOf(best, truth)
                    val work = requireNotNull(optimizer.lastSuggestion).acquisitionEvaluations
                    acquisitionWork += work
                    writer.appendLine("${step + 1},${step + 1},$observed,$truth,${best - 1.0},${optimizer.fitCount},$work," +
                        projection.encode(trial.sample).toDoubleArray().joinToString(";"))
                }
            }
            metrics += listOf(
                BenchMetric("$algorithm.best-regret", best - 1.0, "objective", MetricDirection.LOWER),
                BenchMetric("$algorithm.evaluations", budget.toDouble(), "evaluations"),
                BenchMetric("$algorithm.model-fits", optimizer.fitCount.toDouble(), "fits"),
                BenchMetric("$algorithm.acquisition-evaluations", acquisitionWork.toDouble(), "evaluations"),
                BenchMetric("$algorithm.elapsed", (System.nanoTime() - start) / 1e9, "seconds"),
            )
        }
        return BenchResult(case.id, metrics = metrics, artifacts = context.artifacts())
    }
}
