package com.eignex.combo.bench.diagnostics

import com.eignex.combo.bo.SearchTransform
import com.eignex.combo.decisions.*
import com.eignex.combo.shared.*
import com.eignex.klause.arithmetic.real.exactRealValue
import com.eignex.klause.arithmetic.real.toExactRealValue
import com.eignex.klause.formats.json.ExactRealJson
import com.eignex.klause.formats.json.ProblemJson
import com.eignex.klause.ir.SourceRealId
import com.eignex.klause.schema.eq
import com.eignex.klause.schema.plus
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.Sample
import com.eignex.klause.solver.checkExactRealSample
import com.eignex.klause.util.Cancellation
import kotlin.math.*
import kotlin.random.Random

private class Square : DecisionSpace() {
    val x by floatVar(0.0, 1.0)
    val y by floatVar(0.0, 1.0)
}
private class Equality : DecisionSpace() {
    val x by floatVar(0.0, 1.0)
    val y by floatVar(0.0, 1.0)
    val sum by constraint { (x + y) eq 1.0 }
}
private class LogContext : DecisionSpace() {
    val x by floatVar(1e-6, 1.0, scale = Scale.Log)
    val y by floatVar(0.0, 1.0)
    val load by contextReal(0.0, 1.0)
    val sum by constraint { (x + y) eq load }
}
internal fun exactPoint(values: DoubleArray): Sample = Sample(booleanArrayOf(), longArrayOf(), values,
    exactReals = values.map { BigFraction.ofDouble(it)!! })

internal class DiagnosticProblem(val name: String, val contextLevel: Double = 0.75) {
    val space: CompiledDecisionSpace
    val context: Context
    val optimum = 0.0
    private val exactExpression = if (name == "exact-symbolic") """{"exp":"1"}""" else """{"root":{"coefficients":["-2","0","1"],"index":2}}"""
    init {
        when (name) {
            "equality" -> { space = Equality().compileSpace(); context = Context.Empty }
            "context-log" -> {
                val model = LogContext()
                space = model.compileSpace()
                context = context { set(model.load, contextLevel) }
            }
            "exact-algebraic", "exact-symbolic" -> {
                space = CompiledDecisionSpace.fromProblem(ProblemJson.default.decodeFromString(
                    """{"bools":1,"reals":[{},{"lower":0,"upper":1}]}"""))
                context = context { set(SourceRealId(0), ExactRealJson.decodeFromString(exactExpression)) }
            }
            else -> { require(name in listOf("quadratic2", "branin2")); space = Square().compileSpace(); context = Context.Empty }
        }
    }
    val projection = SearchTransform(space)
    fun point(random: Random): BanditSample = point(DoubleArray(2) { random.nextDouble() })
    fun point(unit: DoubleArray): BanditSample {
        val values = when (name) {
            "equality" -> doubleArrayOf(unit[0], 1.0 - unit[0])
            "context-log" -> {
                val x = exp(ln(1e-6) + unit[0] * (ln(contextLevel) - ln(1e-6)))
                doubleArrayOf(x, contextLevel - x, contextLevel)
            }
            "exact-algebraic", "exact-symbolic" -> doubleArrayOf(if (name == "exact-symbolic") E else sqrt(2.0), unit[0])
            else -> unit
        }
        if (name.startsWith("exact-")) {
            val expression = ExactRealJson.decodeFromString(exactExpression)
            return BanditSample(Sample(booleanArrayOf(unit[1] > 0.5), longArrayOf(), values,
                symbolicReals = listOf(expression, BigFraction.ofDouble(values[1])!!.toExactRealValue())))
        }
        val exact = values.map { BigFraction.ofDouble(it)!! }.toMutableList()
        if (name == "equality") exact[1] = BigFraction.ONE - exact[0]
        if (name == "context-log") exact[1] = exact[2] - exact[0]
        return BanditSample(Sample(booleanArrayOf(), longArrayOf(), values, exactReals = exact))
    }
    fun truth(point: BanditSample): Double {
        val x = point.sample.reals
        return when (name) {
            "branin2" -> {
                val a = 15 * x[0] - 5
                val b = 15 * x[1]
                val residual = b - 5.1 / (4 * PI * PI) * a * a + 5 / PI * a - 6
                maxOf(0.0, (residual * residual + 10 * (1 - 1 / (8 * PI)) * cos(a) + 10 - 0.39788735772973816) / 100)
            }
            "equality" -> (x[0] - 0.37).pow(2)
            "context-log" -> (ln(x[0] / (contextLevel * 0.31)) / ln(1e6)).pow(2)
            "exact-algebraic", "exact-symbolic" -> (x[1] - 0.73).pow(2) + if (point.bools[0]) 0.1 else 0.0
            else -> ((x[0] - 0.25).pow(2) + 2 * (x[1] - 0.33).pow(2)) / 3.0
        }
    }
    fun encoded(points: List<BanditSample>) = matrix(points.map { projection.encode(it, context).toDoubleArray() }, projection.featureSize)
    fun identity(sample: Sample): String = sample.bools.joinToString() + "/" +
        (0 until sample.numIntVars).joinToString { sample.exactIntValue(it).toString() } + "/" +
        (0 until sample.numRealVars).joinToString { ExactRealJson.encodeToString(sample.exactRealValue(it)) }
    fun json(sample: Sample): String = "{\"bools\":[${sample.bools.joinToString()}],\"ints\":[" +
        (0 until sample.numIntVars).joinToString { "\"${sample.exactIntValue(it)}\"" } + "],\"reals\":[" +
        (0 until sample.numRealVars).joinToString { ExactRealJson.encodeToString(sample.exactRealValue(it)) } + "]}"
    fun check(sample: Sample, cancelled: Boolean = false): Boolean? {
        var polls = 0
        return space.problem.checkExactRealSample(sample, Cancellation { cancelled || ++polls > 100_000 || Thread.currentThread().isInterrupted })
    }

    /** Existing Klause repair path, seeded with independently source-checked original-coordinate proposals. */
    fun witnesses(seed: Int, solver: Boolean = false): MeasuredWitnesses = MeasuredWitnesses(this, seed, solver)
}

internal class MeasuredWitnesses(private val problem: DiagnosticProblem, seed: Int, solver: Boolean) : WitnessSearch {
    private val hybrid = if (solver) HybridDescentSession.klause(problem.space, candidateBatchSize = 4, poolCapacity = 64, context = problem.context) else null
    var calls = 0L
    var checks = 0L
    var unavailable = 0L
    var rejected = 0L
    var nanos = 0L
    var generated = 0L
    val terminations = mutableMapOf<String, Int>()
    private val source = Random(seed xor 0x7769746e)
    val proposals: Long get() = hybrid?.proposalCalls ?: 0
    val backtracks: Long get() = hybrid?.backtrackCalls ?: 0
    override fun witnesses(request: WitnessRequest, random: Random): WitnessBatch {
        val start = System.nanoTime()
        check(!Thread.currentThread().isInterrupted) { "benchmark cancelled" }
        calls++
        val anchors = List(minOf(maxOf(request.count, 16), request.maxAttempts)) { problem.point(source).sample }
        generated += anchors.size
        val result = if (hybrid != null) hybrid.witnesses(request.copy(anchors = request.anchors + anchors,
            maxAttempts = request.maxAttempts - anchors.size, context = Context.Empty), random) else {
            val excluded = request.exclude.map { problem.identity(it) }.toMutableSet()
            val points = (request.anchors + anchors).filter { point ->
                if (!problem.space.matches(point, request.context) || !problem.space.matches(point, request.assumptions)) false else {
                    checks++
                    when (problem.check(point)) {
                        true -> excluded.add(problem.identity(point))
                        false -> { rejected++; false }
                        null -> { unavailable++; false }
                    }
                }
            }.take(request.count)
            WitnessBatch(points, if (points.size == request.count) WitnessTermination.CountReached else WitnessTermination.BudgetExhausted)
        }
        terminations.merge(result.termination.name, 1, Int::plus)
        nanos += System.nanoTime() - start
        return result
    }
}
