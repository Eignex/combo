package combo.decisions

import com.eignex.klause.ast.implies
import com.eignex.klause.ast.le
import com.eignex.klause.ast.not
import com.eignex.klause.solver.LocalSearchParams
import com.eignex.klause.solver.LocalSearchSolver
import combo.bandit.glm.ConstantRate
import combo.bandit.glm.DiagonalizedLinearModel
import combo.bandit.glm.LinearBandit
import combo.decisions.BanditSample
import combo.bandit.glm.LinearFeatureProjection
import combo.bandit.glm.NormalVariance
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

private class ContextInConstraint : DecisionSpace() {
    val premium by contextBool()
    val budget  by intVar(0, 4000)
    // Constraint mentions context variable — only legal if contexts are solver-visible.
    val capForNonPremium by constraint { !premium implies (budget le 1000) }
}

private class AgeOptional : DecisionSpace() {
    val age by optionalContextInt(0, 120)
}

private class OptionalCtxInteraction : DecisionSpace() {
    val choice by boolVar()
    val age    by optionalContextInt(0, 120)
    val ageXchoice by interact(age, choice)
}

class OptionalContextTest {

    @Test
    fun `constraint mentioning required context should enforce limit when context is absent of premium`() {
        val model = ContextInConstraint()
        val space = model.compileSpace()
        val projection = LinearFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)

        // premium=false → budget must be ≤ 1000.
        repeat(10) { seed ->
            val ctx = context { set(model.premium, false) }
            val asm = projection.assumptionsFor(ctx)
            val s = solver.sample(LocalSearchParams(randomSeed = seed.toLong(), assumptions = asm))!!
            assertEquals(false, space.decode(model.premium, s))
            assertTrue(space.compiled.decode(model.budget, s) <= 1000,
                "budget should be ≤ 1000 when premium=false; got ${space.compiled.decode(model.budget, s)}")
        }
        // premium=true relaxes the cap — budget may exceed 1000.
        val ctxOn = context { set(model.premium, true) }
        val asmOn = projection.assumptionsFor(ctxOn)
        var sawHigh = false
        for (seed in 1L..20L) {
            val s = solver.sample(LocalSearchParams(randomSeed = seed, assumptions = asmOn))!!
            if (space.compiled.decode(model.budget, s) > 1000) { sawHigh = true; break }
        }
        assertTrue(sawHigh, "premium=true should permit budget > 1000")
    }

    @Test
    fun `optional context should be absent by default and present when set`() {
        val model = AgeOptional()
        val space = model.compileSpace()
        val projection = LinearFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)

        // Absent: isKnown=false pinned via assumption, klause's __pin_age forces age=0.
        val ctxAbsent = context { }
        val asmAbsent = projection.assumptionsFor(ctxAbsent)
        val sAbsent = solver.sample(LocalSearchParams(randomSeed = 1, assumptions = asmAbsent))!!
        assertEquals(0, space.decode(model.age, sAbsent))
        assertFalse(space.isActive(model.age, sAbsent), "age should be inactive when absent")

        // Present: pin to 42.
        val ctxPresent = context { set(model.age, 42) }
        val asmPresent = projection.assumptionsFor(ctxPresent)
        val sPresent = solver.sample(LocalSearchParams(randomSeed = 2, assumptions = asmPresent))!!
        assertEquals(42, space.decode(model.age, sPresent))
        assertTrue(space.isActive(model.age, sPresent))
    }

    @Test
    fun `optional context interaction should be zeroed when context is absent`() {
        val model = OptionalCtxInteraction()
        val space = model.compileSpace()
        val projection = LinearFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)

        // Build a synthetic sample with choice=true. Encode under absent vs present age.
        val params = LocalSearchParams(randomSeed = 1L)
        val sAbsent = solver.sample(params.copy(assumptions = projection.assumptionsFor(context { })))!!
        val fAbsent = projection.encode(BanditSample.undithered(sAbsent), context { })
        // ageXchoice slot is the last one in the layout (numBool + numInt offset).
        // When absent: age is pinned to 0, so age * choice = 0 regardless of choice.
        // Find the interaction slot.
        val interactionSlot = projection.layout.interactionStart.getValue(
            space.interactions.single { it.name == "ageXchoice" }
        )
        assertEquals(0f, fAbsent[interactionSlot],
            "interaction with absent context should be zero")

        // Present, age=30: interaction = 30 * choice_value.
        val ctxPresent = context { set(model.age, 30) }
        val sPresent = solver.sample(params.copy(assumptions = projection.assumptionsFor(ctxPresent)))!!
        val fPresent = projection.encode(BanditSample.undithered(sPresent), ctxPresent)
        val choiceVal = if (space.compiled.decode(model.choice, sPresent)) 1f else 0f
        assertEquals(30f * choiceVal, fPresent[interactionSlot])
    }

    @Test
    fun `bandit with optional context should still choose feasibly when context is absent`() {
        val model = OptionalCtxInteraction()
        val space = model.compileSpace()
        val projection = LinearFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)
        val params = LocalSearchParams(maxFlips = 1_000, randomSeed = 7L)

        val linModel = DiagonalizedLinearModel.Builder(projection.featureSize)
            .family(NormalVariance)
            .learningRate(ConstantRate(1f))
            .priorPrecision(0.01f)
            .exploration(0f)
            .build()
        val bandit = LinearBandit(
            projection = projection,
            linearModel = linModel,
            innerOptimizer = { obj, asm -> solver.minimize(obj, params.copy(assumptions = asm)) },
            randomSeed = 7,
        )

        // Train a few rounds with absent age, just to exercise the path.
        repeat(50) { bandit.update(bandit.chooseOrThrow(context { }), context { }, reward = 0.0) }
        // Then train with present age.
        val ctxPresent = context { set(model.age, 50) }
        repeat(50) { bandit.update(bandit.chooseOrThrow(ctxPresent), ctxPresent, reward = 1.0) }

        // optimal under absent context must produce age=0 (pinned default).
        val outAbsent = bandit.optimalOrThrow(context { })
        assertEquals(0, space.decode(model.age, outAbsent.sample))
        // optimal under present context must produce age=50 (pinned).
        val outPresent = bandit.optimalOrThrow(ctxPresent)
        assertEquals(50, space.decode(model.age, outPresent.sample))
    }
}
