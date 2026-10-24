package com.eignex.combo.bo

import com.eignex.combo.bo.model.BatchForestConfig
import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.decisions.Scale
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.combo.shared.LinearFeatureProjection
import com.eignex.klause.schema.eq
import com.eignex.klause.schema.plus
import java.io.File
import kotlin.math.ln
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** Reproducible comparison with external-evaluation budgets; performance rankings are not test assertions. */
class BoBenchmarkTest {
    private class Quadratic : DecisionSpace() {
        val x by floatVar(0.0, 1.0)
    }
    private class Logarithmic : DecisionSpace() {
        val x by floatVar(1e-6, 1.0, scale = Scale.Log)
    }
    private class Equality : DecisionSpace() {
        val x by floatVar(0.0, 1.0)
        val y by floatVar(0.0, 1.0)
        val sum by constraint { (x + y) eq 1.0 }
    }
    private class Mixed : DecisionSpace() {
        val x by floatVar(0.0, 1.0)
        val count by intVar(1, 6)
        val mode by nominal("slow", "fast")
    }
    private class Noisy : DecisionSpace() {
        val x by floatVar(0.0, 1.0)
        val y by floatVar(0.0, 1.0)
    }
    private class Problem(
        val name: String,
        val space: CompiledDecisionSpace,
        val reference: (BanditSample) -> Double,
        val noise: Double = 0.0,
    )
    private fun quadratic(): Problem {
        val model = Quadratic()
        val space = model.compileSpace()
        return Problem("quadratic", space, { -square(it.float(model.x, space) - 0.73) })
    }
    private fun logarithmic(): Problem {
        val model = Logarithmic()
        val space = model.compileSpace()
        return Problem("logarithmic", space, { -square(ln(it.float(model.x, space) / 1e-3) / ln(1e3)) })
    }
    private fun equality(): Problem {
        val model = Equality()
        val space = model.compileSpace()
        return Problem("equality", space, { -square(it.float(model.x, space) - 0.37) })
    }
    private fun mixed(): Problem {
        val model = Mixed()
        val space = model.compileSpace()
        return Problem("mixed", space, {
            -square(it.float(model.x, space) - 0.65) -
                square((space.compiled.decode(model.count, it.sample) - 4).toDouble() / 6.0) -
                if (space.compiled.decode(model.mode, it.sample) == "fast") 0.0 else 0.15
        })
    }
    private fun noisy(): Problem {
        val model = Noisy()
        val space = model.compileSpace()
        return Problem("noisy", space, {
            -square(it.float(model.x, space) - 0.31) - 3.0 * square(it.float(model.y, space) - 0.72)
        }, noise = 0.05)
    }

    private fun optimizer(problem: Problem, algorithm: String, seed: Int): TrialOptimizer {
        val space = problem.space
        val search = HybridDescentSession.klause(space, candidateBatchSize = 32, poolCapacity = 128)
        if (algorithm == "random") return StudyRunner(Study(space), RandomSuggestionPolicy(space, search, seed))
        val config = ForestOptimizerConfig(
            forest = BatchForestConfig(nbrTrees = 8),
            candidateCount = 32,
            maxWitnessAttempts = 64,
            localSteps = if (algorithm == "local-ei") 6 else 0,
            neighbors = 12,
            repeatPolicy = if (algorithm == "local-ei") RepeatPolicy.NoisyOnly else RepeatPolicy.Allow,
            tieBreak = if (algorithm == "local-ei") AcquisitionTieBreak.Random else AcquisitionTieBreak.First,
        )
        return RandomForestOptimizer(
            projection = if (algorithm == "legacy-pool-ei") LinearFeatureProjection(space) else SearchTransform(space),
            config = config,
            search = search,
            randomSeed = seed,
            outcome = OutcomeSpec(
                noise = if (problem.noise > 0.0) ObservationNoise.Noisy else ObservationNoise.Deterministic,
            ),
        )
    }
    private fun run(problem: Problem, algorithm: String, seed: Int): BenchmarkResult {
        val noise = Random(seed.toLong() xor 0x4142L)
        return OptimizationBenchmark.run(
            optimizer(problem, algorithm, seed),
            budget = 32,
            evaluate = {
                val reference = problem.reference(it)
                val reward = reference + if (problem.noise == 0.0) {
                    0.0
                } else {
                    noise.nextDouble(-problem.noise, problem.noise)
                }
                BenchmarkEvaluation(reward, reference = reference)
            },
            optimum = 0.0,
        )
    }
    private fun compare(factory: () -> Problem) {
        val rows = mutableListOf("problem,algorithm,seed,evaluations,cumulative_cost,best_reference,simple_regret")
        val seeds = listOf(3, 11, 19, 29, 43, 61)
        for (algorithm in listOf("local-ei", "pool-ei", "legacy-pool-ei", "random")) {
            for (seed in seeds) {
                val problem = factory()
                val result = run(problem, algorithm, seed)
                assertEquals(32, result.bestValues.size, "${problem.name} $algorithm seed=$seed")
                assertEquals(32.0, result.cumulativeCost[31])
                val regret = result.simpleRegret!!
                assertTrue(regret.toDoubleArray().all { it.isFinite() && it >= 0.0 })
                assertTrue((1 until regret.size).all { regret[it] <= regret[it - 1] })
                for (i in 0 until result.bestValues.size) {
                    rows += "${problem.name},$algorithm,$seed,${i + 1},${result.cumulativeCost[i]}," +
                        "${result.bestValues[i]},${regret[i]}"
                }
                if (seed == seeds.first()) {
                    val repeated = run(factory(), algorithm, seed)
                    assertEquals(
                        result.bestValues.toDoubleArray().toList(),
                        repeated.bestValues.toDoubleArray().toList(),
                    )
                }
            }
        }
        val directory = File("build/reports/bo-benchmark").also { it.mkdirs() }
        File(directory, "${factory().name}.csv").writeText(rows.joinToString("\n", postfix = "\n"))
    }

    @Test
    fun `quadratic regret compares guided pool-only and random search across seeds`() = compare(::quadratic)

    @Test
    fun `logarithmic regret compares guided pool-only and random search across seeds`() = compare(::logarithmic)

    @Test
    fun `equality constrained regret compares guided pool-only and random search across seeds`() = compare(::equality)

    @Test
    fun `mixed categorical integer regret compares guided pool-only and random search across seeds`() = compare(::mixed)

    @Test
    fun `noisy regret compares guided pool-only and random search using noiseless references`() = compare(::noisy)

    private fun square(value: Double) = value * value
}
