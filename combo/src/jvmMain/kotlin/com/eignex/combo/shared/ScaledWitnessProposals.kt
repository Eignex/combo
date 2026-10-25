package com.eignex.combo.shared

import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.solver.Sample
import kotlin.math.roundToLong
import kotlin.random.Random

internal class ScaledWitnessProposals(private val space: CompiledDecisionSpace) {
    private val validator = WitnessValidator(space)
    private val layout = SolverCoordinateLayout(space)
    private val contexts = space.contextInts.map { it.name }.toSet()

    fun integerPins(assumptions: Assumptions, random: Random): Assumptions {
        val pins = mutableMapOf<Int, Long>()
        for ((name, domain) in space.numericDomains) {
            if (name in contexts) continue
            val id = space.compiled.intVarIdByName[name] ?: continue
            if (id in assumptions.ints) continue
            val value = domain.fromUnit(random.nextDouble())
            val spec = layout.floatSpecs[id]
            pins[id] = when {
                spec == null -> value.roundToLong()
                spec.scale == 0.0 -> 0L
                else -> ((value - spec.min) / spec.scale).roundToLong()
            }
        }
        return if (pins.isEmpty()) assumptions else assumptions.mergedWith(Assumptions(ints = pins))
    }

    fun interior(anchor: Sample, random: Random): Sample {
        // Retain boundary coverage alongside proposals drawn in the declared geometry.
        if (random.nextInt(4) == 0) return anchor
        val reals = anchor.reals
        val exact = (anchor.exactReals ?: reals.map { BigFraction.ofDouble(it) ?: return anchor }).toMutableList()
        var changed = false
        for ((id, handle) in layout.realHandles) {
            if (space.activeConditions[handle.name]?.let { !space.evaluate(it, anchor) } == true) continue
            val domain = space.numericDomains[handle.name] ?: continue
            reals[id] = domain.fromUnit(random.nextDouble())
            exact[id] = BigFraction.ofDouble(reals[id]) ?: return anchor
            changed = true
        }
        if (!changed) return anchor
        val proposed = Sample(anchor.bools, anchor.ints, reals, exactReals = exact, exactInts = anchor.exactInts)
        return if (validator.valid(proposed)) proposed else anchor
    }
}
