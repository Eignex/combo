package com.eignex.combo.bench

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.decisions.Scale
import kotlin.math.ln
import kotlin.math.sin

private class Continuous : DecisionSpace() {
    val x by floatVar(0.0, 1.0)
}

private class Logarithmic : DecisionSpace() {
    val x by floatVar(1e-6, 1.0, scale = Scale.Log)
}

private class Noisy : DecisionSpace() {
    val x by floatVar(0.0, 1.0)
    val y by floatVar(0.0, 1.0)
}

private class Child : DecisionSpace() {
    val size by intVar(1, 6)
}

private class Conditional : DecisionSpace() {
    val x by floatVar(0.0, 1.0)
    val mode by nominal("slow", "fast")
    val child by optionalDecisionSpace(::Child)
}

internal class Workload(model: DecisionSpace, val truth: (BanditSample) -> Double) {
    val space = model.compileSpace()
}

private fun sq(x: Double) = x * x

internal fun workload(name: String): Workload = when (name) {
    "logarithmic" -> {
        val model = Logarithmic()
        val space = model.compileSpace()
        Workload(model) { 1.0 + sq((ln(it.float(model.x, space)) - ln(1e-6)) / ln(1e6) - 0.62) }
    }
    "conditional" -> {
        val model = Conditional()
        val space = model.compileSpace()
        Workload(model) {
            1.0 + sq(it.float(model.x, space) - 0.62) +
                (if (space.compiled.decode(space.gateOf(model.child)!!, it.sample)) {
                    0.02 * sq(space.compiled.decode(model.child.size, it.sample) - 4.0)
                } else 0.3) +
                (if (space.compiled.decode(model.mode, it.sample) == "fast") 0.0 else 0.1)
        }
    }
    "noisy" -> {
        val model = Noisy()
        val space = model.compileSpace()
        Workload(model) {
            1.0 + sq(it.float(model.x, space) - 0.31) + 3.0 * sq(it.float(model.y, space) - 0.72)
        }
    }
    "quadratic", "multimodal" -> {
        val model = Continuous()
        val space = model.compileSpace()
        Workload(model) {
            val x = it.float(model.x, space)
            1.0 + if (name == "multimodal") {
                0.2 * sq(x - 0.31) + 0.1 * sq(sin(6.0 * kotlin.math.PI * (x - 0.31)))
            } else sq(x - 0.73)
        }
    }
    else -> error("Unknown workload: $name")
}
