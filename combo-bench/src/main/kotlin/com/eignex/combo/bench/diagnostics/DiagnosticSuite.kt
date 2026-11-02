package com.eignex.combo.bench.diagnostics

import com.eignex.combo.bo.*
import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.shared.*
import com.eignex.koblas.koblas
import com.eignex.kumulant.math.nextNormal
import com.eignex.kumulant.schema.ops.mseLoss
import com.eignex.kumulant.schema.runtime.materialize
import com.eignex.kumulant.stat.summary.MeanStat
import com.eignex.lab.bench.*
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest
import java.util.zip.GZIPInputStream
import kotlin.io.path.*
import kotlin.math.*
import kotlin.random.Random

internal const val HISTORY_SHA = "e51cc99f441a616c1f24272555026ec93383e6aef1fef118fde6c2bf28e29d09"
internal const val HISTORY_PATH = "combo-bench/build/fixtures/gp-fitting-validation.csv.gz"
internal fun digest(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
internal fun fileDigest(path: Path): String = Files.newInputStream(path).use { input ->
    val hash = MessageDigest.getInstance("SHA-256")
    val buffer = ByteArray(65536)
    while (true) { val n = input.read(buffer); if (n < 0) break; hash.update(buffer, 0, n) }
    hash.digest().joinToString("") { "%02x".format(it) }
}
internal fun runtimeInventory(): String = System.getProperty("java.class.path").split(java.io.File.pathSeparator)
    .map { Path.of(it) }.filter { it.isRegularFile() }.sortedBy { it.fileName.toString() }
    .joinToString("\n") { "${it.fileName} ${fileDigest(it)}" }

internal fun backendIdentity(): String {
    val getter = koblas.javaClass.methods.firstOrNull { it.name == "getVendor" && it.parameterCount == 0 }
    val vendor = getter?.invoke(koblas) ?: return "vector=${koblas.vectorKernels.name};vendor=none"
    fun property(name: String): String = getter.returnType.methods.firstOrNull { it.name == name && it.parameterCount == 0 }
        ?.invoke(vendor)?.toString() ?: "unreported"
    val library = property("getLibraryPath")
    val hash = runCatching { Path.of(library).takeIf { it.isRegularFile() }?.let(::fileDigest) }.getOrNull() ?: "unavailable"
    return "vector=${koblas.vectorKernels.name};vendor=${property("getVendor")};version=${property("getVersion")};library=$library;sha256=$hash;threads=${property("getThreadEvidence")}"
}

private data class FixtureRow(val kind: String, val point: BanditSample, val observed: Double, val truth: Double)

internal object DiagnosticSuite {
    private val options = setOf("study", "phase", "workloads", "seeds", "repetitions", "budget", "batch", "noise-sd",
        "replicates", "pending", "arms", "fit-trials", "grid-nodes", "probes", "draws", "context", "history-policy", "fit-seed", "smoke", "timeout-seconds", "memory-mib")
    private val studies = setOf("fitting", "noise", "uncertainty", "constrained")
    private fun selected(value: String): List<String> = value.split(',').also { require(it.isNotEmpty() && it.all(String::isNotBlank) && it.distinct().size == it.size) }
    fun plan(request: BenchRequest): List<BenchCase> {
        val p = request.parameters
        require(p.keys.all { it in options }) { "Unknown diagnostic options: ${p.keys - options}" }
        val study = p.getValue("study").also { require(it in studies) }
        val smoke = p["smoke"]?.toBooleanStrict() ?: false
        val phase = p["phase"] ?: "loop"
        require(phase in setOf("loop", "frozen", "historical"))
        require(phase != "historical" || study == "fitting")
        val names = selected(p["workloads"] ?: if (study == "constrained") "equality,context-log,exact-algebraic,exact-symbolic" else "quadratic2,branin2")
        require(names.all { it in if (study == "constrained") setOf("equality", "context-log", "exact-algebraic", "exact-symbolic") else setOf("quadratic2", "branin2") })
        val seeds = selected(p["seeds"] ?: if (phase == "historical") "1901,1902,1903,1904,1905,1906" else if (smoke) "7" else "2101,2102,2103,2104,2105,2106")
            .map { it.toInt() }
        require(seeds.distinct().size == seeds.size)
        fun integer(name: String, default: Int, minimum: Int = 1) = (p[name]?.toInt() ?: default).also { require(it >= minimum) { "$name below $minimum" } }
        val budget = integer("budget", if (smoke) 8 else 32, 4)
        val probes = integer("probes", if (smoke) 8 else 256, 4)
        val draws = integer("draws", if (smoke) 8 else 256, 2).also { require(it % 2 == 0) }
        val nodes = integer("grid-nodes", 3).also { require(it in listOf(3, 5)) }
        val trials = integer("fit-trials", 96, 0)
        val arms = selected(p["arms"] ?: when (study) {
            "fitting" -> "baseline24,coordinate,mean,prior,multistart,combined"
            "noise" -> "known,unknown,partial"
            "uncertainty" -> "coordinate,grid-map,mixture"
            else -> "coordinate,random"
        })
        require(arms.all { it in when (study) {
            "fitting" -> setOf("baseline24", "coordinate", "mean", "prior", "multistart", "combined")
            "noise" -> setOf("known", "unknown", "partial")
            "uncertainty" -> setOf("coordinate", "grid-map", "mixture")
            else -> setOf("coordinate", "combined", "random")
        } })
        val batches = selected(p["batch"] ?: if (smoke) "1" else "1,4").map { it.toInt().also { q -> require(q in listOf(1, 4)) } }
        val deviations = selected(p["noise-sd"] ?: if (study == "noise") "0.0001,0.03,0.15" else if (study == "constrained") "0.0" else "0.0,0.03")
            .map { it.toDouble().also { x -> require(x.isFinite() && x >= 0.0) } }
        val replicates = selected(p["replicates"] ?: "1").map { it.toInt().also { r -> require(r in 1..4) } }
        val pendings = selected(p["pending"] ?: "false").map { it.toBooleanStrict() }
        val contexts = selected(p["context"] ?: "0.75").map { it.toDouble().also { x -> require(x.isFinite() && x in 0.1..1.0) } }
        val fitSeeds = selected(p["fit-seed"] ?: "0").map(String::toInt)
        val historyPolicies = selected(p["history-policy"] ?: "baseline24")
        require(historyPolicies.all { it in listOf("baseline24", "baseline96", "combined96") })
        val resources = BenchResources(1, integer("memory-mib", 2048, 2048).toLong(), integer("timeout-seconds", 900).toLong())
        val inventory = digest(runtimeInventory().toByteArray())
        val result = mutableListOf<BenchCase>()
        for (name in names) for (context in if (name == "context-log") contexts else contexts.take(1))
            for (historyPolicy in historyPolicies) for (fitSeed in fitSeeds)
            for (seed in seeds) for (q in batches) for (sd in deviations) for (r in replicates) for (pending in pendings) {
            require(budget >= q * r) { "budget must cover one complete batch and replicate group" }
            require(phase != "historical" || (seed in 1901..1906 && sd in listOf(0.0, 0.03) && budget <= 32 && r == 1))
            val fixture = fixture(name, seed, sd, r, probes, budget, context)
            val path = "combo-bench/build/diagnostic-inputs/${digest(fixture.toByteArray())}.csv"
            Path.of(path).parent.createDirectories()
            Path.of(path).writeText(fixture)
            val inputs = mutableListOf(BenchInput(path, role = "frozen-history-and-probes", sha256 = digest(fixture.toByteArray())))
            if (phase == "historical") {
                require(Path.of(HISTORY_PATH).isRegularFile() && fileDigest(Path.of(HISTORY_PATH)) == HISTORY_SHA) { "Prepare the pinned historical fixture before planning" }
                inputs += BenchInput(HISTORY_PATH, role = "historical-fitting-traces", sha256 = HISTORY_SHA)
            }
            for (repetition in 1..integer("repetitions", 1)) {
                val parameters = mapOf("study" to study, "phase" to phase, "workload" to name, "seed" to "$seed", "batch" to "$q",
                    "noise-sd" to "$sd", "replicates" to "$r", "pending" to "$pending", "budget" to "$budget", "probes" to "$probes",
                    "arms" to arms.joinToString(","), "fit-trials" to "$trials", "grid-nodes" to "$nodes", "draws" to "$draws", "context" to "$context",
                    "history-policy" to historyPolicy, "fit-seed" to "$fitSeed", "repetition" to "$repetition")
                val id = "$study/$phase/$name/$seed/q$q/sd$sd/r$r/p$pending/$historyPolicy/ctx$context/fit$fitSeed/$repetition"
                result += BenchCase(id, parameters, resources, inputs = inputs,
                    attributes = mapOf("suite-version" to "diagnostics-v1", "runtime-sha256" to inventory,
                        "parameters-sha256" to digest(parameters.toSortedMap().toString().toByteArray())))
            }
        }
        return result
    }
    private fun fixture(name: String, seed: Int, sd: Double, replicates: Int, probes: Int, budget: Int, context: Double): String {
        val problem = DiagnosticProblem(name, context)
        val design = Random(seed xor 0x64657369)
        val noise = Random(seed xor 0x6e6f6973)
        return buildString {
            appendLine("kind,u,v,observed,truth")
            var unit = DoubleArray(2)
            for (i in 0 until budget) {
                if (i % replicates == 0) unit = DoubleArray(2) { design.nextDouble() }
                val truth = problem.truth(problem.point(unit))
                appendLine("train,${unit[0]},${unit[1]},${truth + sd * noise.nextNormal()},$truth")
            }
            val heldout = Random(seed xor 0x70726f62)
            for (i in 0 until probes) {
                val x = DoubleArray(2) { heldout.nextDouble() }
                val truth = problem.truth(problem.point(x))
                appendLine("probe,${x[0]},${x[1]},${truth + sd * noise.nextNormal()},$truth")
            }
        }
    }
    fun validate(case: BenchCase, context: BenchContext): List<Path> {
        require(case.attributes["suite-version"] == "diagnostics-v1")
        val p = case.parameters
        require(case.id == "${p["study"]}/${p["phase"]}/${p["workload"]}/${p["seed"]}/q${p["batch"]}/sd${p["noise-sd"]}/r${p["replicates"]}/p${p["pending"]}/${p["history-policy"]}/ctx${p["context"]}/fit${p["fit-seed"]}/${p["repetition"]}")
        require(case.attributes["parameters-sha256"] == digest(case.parameters.toSortedMap().toString().toByteArray())) { "Frozen parameters changed" }
        require(case.attributes["runtime-sha256"] == digest(runtimeInventory().toByteArray())) { "Resolved runtime artifacts changed" }
        require(case.resources.cpus == 1 && case.resources.memoryMiB >= 2048 && case.resources.timeoutSeconds > 0)
        require(case.inputs.size == if (case.parameters["phase"] == "historical") 2 else 1)
        val root = (context.worktree ?: Path.of("")).toAbsolutePath().normalize()
        return case.inputs.map { input ->
            require(input.root == BenchInputRoot.SOURCE && input.sha256 != null)
            val path = root.resolve(input.path).normalize()
            require(path.startsWith(root) && path.toRealPath().startsWith(root.toRealPath()) && fileDigest(path) == input.sha256) { "Prepared input changed or escaped its source root" }
            path
        }
    }
    fun run(case: BenchCase, context: BenchContext): BenchResult {
        val inputs = validate(case, context)
        val p = case.parameters
        val problem = DiagnosticProblem(p.getValue("workload"), p.getValue("context").toDouble())
        val rows = inputs.first().readLines().drop(1).map { line ->
            val values = line.split(',')
            FixtureRow(values[0], problem.point(doubleArrayOf(values[1].toDouble(), values[2].toDouble())), values[3].toDouble(), values[4].toDouble())
        }
        val frozen = if (p["phase"] == "historical") historical(inputs[1], problem, p) else rows.filter { it.kind == "train" }
        val probes = rows.filter { it.kind == "probe" }
        val report = Files.createDirectory(context.reportPath())
        report.resolve("runtime.txt").writeText("java=${System.getProperty("java.runtime.version")}\nvm=${System.getProperty("java.vm.name")}\n" +
            "os=${System.getProperty("os.name")}/${System.getProperty("os.arch")}\nbackend=${backendIdentity()}\nparameters=$p\n" + runtimeInventory())
        report.resolve("problem.json").writeText(com.eignex.klause.formats.json.ProblemJson.default.encodeToString(problem.space.problem))
        val arms = p.getValue("arms").split(',')
        val offset = (p.getValue("repetition").toInt() - 1) % arms.size
        val metrics = mutableListOf<BenchMetric>()
        try {
            for (arm in arms.drop(offset) + arms.take(offset)) {
                check(!Thread.currentThread().isInterrupted) { "benchmark cancelled" }
                metrics += runArm(case, problem, arm, frozen, probes, report)
            }
        } catch (failure: Exception) {
            val reason = failure.message ?: failure.javaClass.name
            report.resolve("failure.txt").writeText("status=failed\nreason=$reason\n")
            return BenchResult(case.id, BenchStatus.FAILED, metrics, context.artifacts(), message = reason)
        }
        return BenchResult(case.id, metrics = metrics, artifacts = context.artifacts())
    }
    private fun historical(path: Path, problem: DiagnosticProblem, p: Map<String, String>): List<FixtureRow> =
        GZIPInputStream(Files.newInputStream(path)).bufferedReader().use { reader -> reader.lineSequence().drop(1).map { it.split(',') }
            .filter { it[0] == problem.name && it[1] == p["seed"] && it[2].toDouble() == p.getValue("noise-sd").toDouble() &&
                it[3] == p["batch"] && it[4] == p["history-policy"] }
            .take(p.getValue("budget").toInt()).map { row ->
                val point = problem.point(row[9].split(';').map(String::toDouble).toDoubleArray())
                check(abs(problem.truth(point) - row[8].toDouble()) < 1e-12) { "Historical objective definition changed" }
                FixtureRow("train", point, row[7].toDouble(), row[8].toDouble())
            }.toList().also { require(it.size == p.getValue("budget").toInt()) { "Historical history is incomplete" } }
        }
    private fun runArm(case: BenchCase, problem: DiagnosticProblem, arm: String, frozen: List<FixtureRow>, probes: List<FixtureRow>, report: Path): List<BenchMetric> {
        val p = case.parameters
        val seed = p.getValue("seed").toInt()
        val sd = p.getValue("noise-sd").toDouble()
        val replicates = p.getValue("replicates").toInt()
        val budget = p.getValue("budget").toInt()
        val q = p.getValue("batch").toInt()
        val unknown = arm in listOf("unknown", "partial") || (p["study"] == "uncertainty" && sd > 0.0)
        val grid = p.getValue("grid-nodes").toInt()
        val trials = if (arm == "baseline24") 24 else if (p["study"] == "uncertainty")
            grid.toDouble().pow(problem.projection.featureSize + 1 + if (unknown) 1 else 0).toInt() - 1 else p.getValue("fit-trials").toInt()
        val fitter = DiagnosticSurrogate(if (arm in listOf("known", "unknown", "partial", "baseline24", "random")) "coordinate" else arm, trials, unknown, grid, p.getValue("fit-seed").toInt())
        val witnesses = problem.witnesses(seed, p["study"] == "constrained")
        val search = LocalAndRandomSearch(problem.projection, witnesses, candidateCount = 16, localSteps = 4, neighbors = 6,
            localPolicy = LocalSearchPolicy(starts = 2))
        val optimizer = BayesianOptimizer(problem.projection, fitter,
            acquisitionOptimizer = search,
            batchPolicy = JointBatchPolicy(fitter.jointFactory(sd > 0.0, p.getValue("draws").toInt()),
                JointLocalAndRandomSearch(problem.projection, witnesses, candidateCount = 16, passes = 1,
                    localSteps = 4, neighbors = 6, localPolicy = LocalSearchPolicy(starts = 2))),
            initialDesign = InitialDesign { _, _ -> arm == "random" },
            exploration = witnesses,
            outcome = OutcomeSpec(maximize = false, noise = if (sd > 0.0) ObservationNoise.Noisy else ObservationNoise.Deterministic),
            repeatPolicy = RepeatPolicy.Never, randomSeed = seed)
        val prefix = report.resolve(arm)
        val noiseStream = Random(seed xor 0x6e6f6973)
        var evaluated = 0
        var best = Double.POSITIVE_INFINITY
        var acquisitionWork = 0L
        var validations = 0L
        var maximumPending = 0
        val start = System.nanoTime()
        fun supplied(index: Int): Double? = if (arm == "unknown" || (p["study"] == "uncertainty" && sd > 0.0) ||
            (arm == "partial" && index % 2 != 0)) null else sd * sd
        prefix.resolveSibling("$arm-trace.csv").bufferedWriter().use { writer ->
            writer.appendLine("step,observed,truth,best_regret,noise_supplied,coordinates,exact_witness")
            fun record(row: FixtureRow, trial: Trial? = null) {
                check(!Thread.currentThread().isInterrupted) { "benchmark cancelled" }
                validations++
                check(problem.space.matches(row.point.sample, problem.context) && problem.check(row.point.sample) == true) { "Unverified original-coordinate witness" }
                val noise = supplied(evaluated)
                if (trial != null) optimizer.complete(trial.id, row.observed, noiseVariance = noise)
                else optimizer.train(row.point, problem.context, row.observed, noiseVariance = noise)
                evaluated++
                best = minOf(best, row.truth)
                writer.appendLine("$evaluated,${row.observed},${row.truth},$best,${noise ?: "unknown"}," +
                    problem.projection.encode(row.point, problem.context).toDoubleArray().joinToString(";") + ",\"" + problem.json(row.point.sample).replace("\"", "\"\"") + "\"")
            }
            val initial = if (p["phase"] == "loop") minOf(budget, maxOf(4, 8 / replicates * replicates)) else budget
            frozen.take(initial).forEach { record(it); noiseStream.nextNormal() }
            if (p["phase"] == "loop") {
                val queue = ArrayDeque<Trial>()
                while (evaluated < budget) {
                    val slots = (budget - evaluated + replicates - 1) / replicates
                    val pipeline = if (p["pending"] == "true") 2 else 1
                    while (queue.size < minOf(slots, pipeline * q)) {
                        val count = minOf(q, slots - queue.size)
                        val suggested = optimizer.suggest(count, problem.context)
                        check(suggested.size == count) { "Incomplete budget: ${suggested.termination}; remaining=${budget - evaluated}" }
                        acquisitionWork += optimizer.lastSuggestion!!.acquisitionEvaluations
                        queue.addAll(suggested.trials)
                        maximumPending = maxOf(maximumPending, optimizer.pendingTrials.size)
                    }
                    val completed = List(minOf(q, queue.size)) { queue.removeFirst() }.reversed()
                    for (trial in completed) repeat(minOf(replicates, budget - evaluated)) { repeat ->
                        val truth = problem.truth(trial.sample)
                        record(FixtureRow("train", trial.sample, truth + sd * noiseStream.nextNormal(), truth), if (repeat == 0) trial else null)
                    }
                }
                check(queue.isEmpty() && optimizer.pendingTrials.isEmpty()) { "Unaccounted pending evaluations" }
            }
        }
        check(evaluated == budget)
        val recommendation = if (arm == "random") optimizer.bestObserved(problem.context)!!.sample else optimizer.recommendOrThrow(
            LocalAndRandomSearch(problem.projection, problem.witnesses(seed xor 771), candidateCount = 128, localSteps = 4, neighbors = 8), problem.context)
        val recommendationRegret = problem.truth(recommendation)
        check(problem.space.matches(recommendation.sample, problem.context) && problem.check(recommendation.sample) == true)
        val metrics = mutableListOf<BenchMetric>()
        fun metric(name: String, value: Double, unit: String = "objective", direction: MetricDirection = MetricDirection.LOWER) {
            metrics += BenchMetric("$arm.$name", value, unit, direction)
        }
        metric("best-regret", best)
        metric("recommendation-regret", recommendationRegret)
        metric("evaluations", evaluated.toDouble(), "evaluations", MetricDirection.NONE)
        metric("model-fits", optimizer.fitCount.toDouble(), "fits", MetricDirection.NONE)
        metric("fit-candidates", fitter.candidates.toDouble(), "candidates", MetricDirection.NONE)
        metric("fit-seconds", fitter.fitNanos / 1e9, "seconds")
        metric("acquisition-evaluations", acquisitionWork.toDouble(), "evaluations", MetricDirection.NONE)
        metric("max-pending", maximumPending.toDouble(), "trials", MetricDirection.NONE)
        metric("witness-requests", witnesses.calls.toDouble(), "calls", MetricDirection.NONE)
        metric("solver-proposals", witnesses.proposals.toDouble(), "calls", MetricDirection.NONE)
        metric("generated-anchors", witnesses.generated.toDouble(), "proposals", MetricDirection.NONE)
        metric("solver-backtracks", witnesses.backtracks.toDouble(), "calls", MetricDirection.NONE)
        metric("completed-source-checks", validations.toDouble(), "checks", MetricDirection.NONE)
        metric("witness-seconds", witnesses.nanos / 1e9, "seconds")
        metric("loop-seconds", (System.nanoTime() - start) / 1e9, "seconds")
        report.resolve("$arm-diagnostics.txt").writeText("termination-counts=${witnesses.terminations}\n" +
            "recommendation=${problem.json(recommendation.sample)}\nnoise=${fitter.selectedNoise}\nlikelihood=${fitter.likelihood}\n" +
            "fit-score=${fitter.fitScore}\njitter=${fitter.numericalJitter}\nparameters=${fitter.parameters}\n" +
            "instrumentation-scope=outer witness callbacks and explicit completed-source checks; not all internal validation calls\n")
        if (arm != "random") {
            metric("noise-variance", fitter.selectedNoise, "variance", MetricDirection.NONE)
            metric("noise-absolute-error", abs(fitter.selectedNoise - sd * sd), "variance")
            val model = fitter.last
            val features = problem.encoded(probes.map { it.point })
            val components = fitter.components()
            val predictions = components.map { it.first.predict(features) }
            val componentNoise = components.map { (model, _) ->
                val base = (model as? AffinePosterior)?.base
                if (unknown && base is IntegratedMeanGp) base.noise * (model as AffinePosterior).scale.pow(2)
                else if (unknown) fitter.selectedNoise else sd * sd
            }
            val mse = mseLoss().materialize()
            val coverage = MeanStat()
            val observationCoverage = MeanStat()
            val latentNll = MeanStat()
            val observationNll = MeanStat()
            val total = model.predict(features)
            val stream = Random(seed xor 0x63616c69)
            val uniforms = DoubleArray(2048) { stream.nextDouble() }
            val normals = DoubleArray(2048) { stream.nextNormal() }
            report.resolve("$arm-predictions.csv").bufferedWriter().use { writer ->
                writer.appendLine("truth,observation,mean,latent_variance,latent_lo,latent_hi,observation_lo,observation_hi")
                for (i in probes.indices) {
                    val latent = DoubleArray(normals.size)
                    val observed = DoubleArray(normals.size)
                    val weights = components.map { it.second }.toDoubleArray()
                    var density = 0.0
                    var observedDensity = 0.0
                    for (j in predictions.indices) {
                        val variance = maxOf(1e-15, predictions[j].variances[i])
                        val m = predictions[j].means[i]
                        density += weights[j] * exp(-0.5 * (probes[i].truth - m).pow(2) / variance) / sqrt(2 * PI * variance)
                        observedDensity += weights[j] * exp(-0.5 * (probes[i].observed - m).pow(2) / (variance + componentNoise[j])) / sqrt(2 * PI * (variance + componentNoise[j]))
                    }
                    for (draw in normals.indices) {
                        val j = choose(weights, uniforms[draw])
                        latent[draw] = predictions[j].means[i] + sqrt(maxOf(0.0, predictions[j].variances[i])) * normals[draw]
                        observed[draw] = predictions[j].means[i] + sqrt(maxOf(0.0, predictions[j].variances[i] + componentNoise[j])) * normals[draw]
                    }
                    latent.sort(); observed.sort()
                    val lo = latent[51]; val hi = latent[1996]
                    val olo = observed[51]; val ohi = observed[1996]
                    coverage.update(if (probes[i].truth in lo..hi) 1.0 else 0.0)
                    observationCoverage.update(if (probes[i].observed in olo..ohi) 1.0 else 0.0)
                    latentNll.update(-ln(maxOf(1e-300, density)))
                    observationNll.update(-ln(maxOf(1e-300, observedDensity)))
                    mse.update(total.means[i], probes[i].truth)
                    writer.appendLine("${probes[i].truth},${probes[i].observed},${total.means[i]},${total.variances[i]},$lo,$hi,$olo,$ohi")
                }
            }
            metric("rmse", sqrt(mse.read().mean))
            metric("latent-coverage95", coverage.read().mean, "fraction", MetricDirection.NONE)
            metric("observation-coverage95", observationCoverage.read().mean, "fraction", MetricDirection.NONE)
            metric("latent-nll", latentNll.read().mean, "nats")
            metric("observation-nll", observationNll.read().mean, "nats")
            metric("recommendation-pool-mean-gap", maxOf(0.0, optimizer.predict(recommendation, problem.context) - total.means.toDoubleArray().min()))
        }
        return metrics
    }
}
