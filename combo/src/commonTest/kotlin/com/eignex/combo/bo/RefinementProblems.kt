package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.decisions.SubSpace
import com.eignex.combo.shared.WitnessBatch
import com.eignex.combo.shared.WitnessSearch
import com.eignex.combo.shared.WitnessTermination
import com.eignex.combo.shared.assignmentKey
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.Sample
import kotlin.math.floor
import kotlin.math.sin

private class RefinementContinuous : DecisionSpace() {
    val x by floatVar(0.0, 1.0)
}
private class RefinementLogarithmic : DecisionSpace() {
    val x by floatVar(1e-6, 1.0, scale = com.eignex.combo.decisions.Scale.Log)
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
internal class RefinementProblem(val name: String, val model: DecisionSpace, val truth: (BanditSample) -> Double) {
    val space = model.compileSpace()
}
private fun sq(x: Double) = x * x
internal fun refinementProblem(name: String): RefinementProblem {
    if (name == "logarithmic") {
        val model = RefinementLogarithmic()
        val space = model.compileSpace()
        return RefinementProblem(name, model) {
            1.0 + sq((kotlin.math.ln(it.float(model.x, space)) - kotlin.math.ln(1e-6)) / kotlin.math.ln(1e6) - 0.62)
        }
    }
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
        1.0 + if (name == "multimodal") {
            0.2 * sq(x - 0.31) + 0.1 * sq(sin(6.0 * kotlin.math.PI * (x - 0.31)))
        } else {
            sq(x - 0.73)
        }
    }
}
internal fun measured(problem: RefinementProblem, point: BanditSample, key: Int): Double {
    val source = point.sample.reals
    val hash = sin(key * 12.9898 + source[0] * 78.233 + source.getOrElse(1) { 0.0 } * 39.425) * 43758.5453
    return problem.truth(point) + if (problem.name == "noisy") 0.05 * (2 * (hash - floor(hash)) - 1) else 0.0
}
internal fun continuousPoint(values: DoubleArray): BanditSample = BanditSample(
    Sample(
        booleanArrayOf(),
        longArrayOf(),
        values,
        exactReals = values.map { BigFraction.ofDouble(it)!! },
    ),
)

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
