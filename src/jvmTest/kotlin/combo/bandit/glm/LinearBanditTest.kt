package combo.bandit.glm

import com.eignex.klause.solver.localsearch.LocalSearchParams
import com.eignex.klause.solver.localsearch.LocalSearchSolver
import com.eignex.klause.solver.Sample
import com.eignex.kumulant.bandit.FactorisedGaussian
import com.eignex.kumulant.stat.regression.ConstantRate
import com.eignex.kumulant.stat.regression.DiagonalRegression
import combo.decisions.DecisionSpace
import combo.decisions.SubSpace
import combo.decisions.context
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

private class FiveToggles : DecisionSpace() {
    val a by boolVar()
    val b by boolVar()
    val c by boolVar()
    val d by boolVar()
    val e by boolVar()
}

private class TogglesWithContext : DecisionSpace() {
    val choice1 by boolVar()
    val choice2 by boolVar()
    val premium by contextBool()
    val segment by contextInt(-100, 100)
}

private class AudioBlock : SubSpace() {
    val mute by boolVar()
}

private class WithOptional : DecisionSpace() {
    val baseline by boolVar()
    val audio by optionalDecisionSpace(::AudioBlock)
}

private class WithNominal : DecisionSpace() {
    val type by nominal("a", "b", "c")
}

private class NominalInteraction : DecisionSpace() {
    val type by nominal("a", "b", "c")
    val premium by contextBool()
    val typeXpremium by interact(premium, type)
}

private class ContextConditionedSchema : DecisionSpace() {
    val choice1 by boolVar()
    val choice2 by boolVar()
    val premium by contextBool()
    val segment by contextInt(-100, 100)
    val premium_x_c1 by interact(premium, choice1)
    val segment_x_c2 by interact(segment, choice2)
}

/** Construct the canonical bandit used by these tests: diagonal-precision regression
 *  + factorised Gaussian sampler. Hyperparameters match what the legacy
 *  `DiagonalizedLinearModel.Builder` used to default to. */
private fun diagonalBandit(
    projection: LinearFeatureProjection,
    solver: LocalSearchSolver,
    params: LocalSearchParams,
    exploration: Double,
    priorPrecision: Double = 0.01,
    learningRateEta: Double = 1.0,
    randomSeed: Int,
) = LinearBandit(
    projection = projection,
    regression = DiagonalRegression(
        featureSize = projection.featureSize,
        priorPrecision = priorPrecision,
        learningRate = ConstantRate(learningRateEta),
    ),
    posterior = FactorisedGaussian,
    exploration = exploration,
    innerOptimizer = { obj, asm -> solver.minimize(obj, params.copy(assumptions = asm)) },
    randomSeed = randomSeed,
)

class LinearBanditTest {

    @Test
    fun `thompson sampling should converge to best sample with no context`() {
        val space = FiveToggles().compileSpace()
        val projection = LinearFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)
        val params = LocalSearchParams(maxFlips = 1_000L, randomSeed = 11L)
        val bandit = diagonalBandit(projection, solver, params, exploration = 0.3, randomSeed = 31)

        // Reward = Σ w_i * bool_i (raw 0/1 features). Best arms: a, c, e set; b, d unset.
        val trueWeights = doubleArrayOf(+1.0, -1.0, +1.0, -1.0, +1.0)
        fun groundTruth(sample: combo.decisions.BanditSample): Double {
            var r = 0.0
            for (i in 0 until 5) r += trueWeights[i] * (if (sample.bools[i]) 1.0 else 0.0)
            return r
        }
        val optimalReward = trueWeights.filter { it > 0 }.sum()  // = 3.0

        val rng = Random(31)
        var earlyRegret = 0.0
        var lateRegret = 0.0
        val rounds = 1500
        repeat(rounds) { i ->
            val sample = bandit.chooseOrThrow()
            val reward = groundTruth(sample) + rng.nextGaussian() * 0.05
            val r = optimalReward - groundTruth(sample)
            if (i < rounds / 3) earlyRegret += r else if (i >= rounds * 2 / 3) lateRegret += r
            bandit.update(sample, reward)
        }
        val earlyAvg = earlyRegret / (rounds / 3)
        val lateAvg = lateRegret / (rounds / 3)
        assertTrue(lateAvg < earlyAvg, "Bandit should improve: early=$earlyAvg, late=$lateAvg")
        val bestReward = groundTruth(bandit.optimalOrThrow())
        assertTrue(bestReward >= optimalReward - 1.0,
            "optimal sample reward $bestReward should be within 1.0 of $optimalReward")
    }

    @Test
    fun `fixed context should steer bandit`() {
        val schema = TogglesWithContext()
        val space = schema.compileSpace()
        val projection = LinearFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)
        val params = LocalSearchParams(maxFlips = 1_000L, randomSeed = 17L)
        val bandit = diagonalBandit(projection, solver, params, exploration = 0.0, randomSeed = 17)

        val ctx = context {
            set(schema.premium, true)
            set(schema.segment, 1)
        }
        fun groundTruth(s: combo.decisions.BanditSample): Double {
            return (if (s.bools[0]) 1.0 else 0.0) + (if (s.bools[1]) 1.0 else 0.0)
        }

        val rng = Random(17)
        repeat(800) {
            val s = combo.decisions.BanditSample.undithered(solver.sample(params.copy(randomSeed = rng.nextLong()))!!)
            bandit.train(s, ctx, groundTruth(s))
        }

        val best = bandit.optimalOrThrow(ctx)
        assertTrue(best.bools[0], "choice1 should be picked under positive-reward training")
        assertTrue(best.bools[1], "choice2 should be picked under positive-reward training")
    }

    @Test
    fun `nominal interaction should expand per label`() {
        val schema = NominalInteraction()
        val space = schema.compileSpace()
        val projection = LinearFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)
        val params = LocalSearchParams(maxFlips = 1_000L, randomSeed = 9L)

        // Layout: 3 nominal indicators + 1 context bool + 3 interaction slots (per label).
        assertEquals(7, projection.featureSize)

        val bandit = diagonalBandit(projection, solver, params, exploration = 0.0, randomSeed = 9)

        fun groundTruth(type: String, premium: Boolean): Double = when {
            premium && type == "b" -> 1.0
            !premium && type == "c" -> 1.0
            else -> 0.0
        }

        val rng = Random(9)
        val ctxTrue = context { set(schema.premium, true) }
        val ctxFalse = context { set(schema.premium, false) }
        repeat(2000) {
            val ctx = if (rng.nextBoolean()) ctxTrue else ctxFalse
            val premium = ctx === ctxTrue
            val s = combo.decisions.BanditSample.undithered(solver.sample(params.copy(randomSeed = rng.nextLong()))!!)
            val type = space.compiled.decode(schema.type, s.sample)
            bandit.train(s, ctx, groundTruth(type, premium))
        }

        assertEquals("b", space.compiled.decode(schema.type, bandit.optimalOrThrow(ctxTrue).sample))
        assertEquals("c", space.compiled.decode(schema.type, bandit.optimalOrThrow(ctxFalse).sample))
    }

    @Test
    fun `nominal decisions should expand to per-label features`() {
        val schema = WithNominal()
        val space = schema.compileSpace()
        val projection = LinearFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)
        val params = LocalSearchParams(maxFlips = 1_000L, randomSeed = 123L)

        assertEquals(3, space.compiled.problem.numBoolVars)
        assertEquals(3, projection.featureSize)
        val indicators = space.compiled.nominalIndicators[schema.type.name]!!
        assertEquals(setOf("a", "b", "c"), indicators.keys)

        val labelReward = mapOf("a" to 0.0, "b" to 1.0, "c" to 0.2)
        val bandit = diagonalBandit(projection, solver, params, exploration = 0.1, randomSeed = 123)
        val rng = Random(123)
        repeat(800) {
            val sample = bandit.chooseOrThrow()
            val picked = space.compiled.decode(schema.type, sample.sample)
            val noisy = labelReward[picked]!! + rng.nextGaussian() * 0.05
            bandit.update(sample, noisy)
        }
        val best = bandit.optimalOrThrow()
        assertEquals("b", space.compiled.decode(schema.type, best.sample))
    }

    @Test
    fun `interactions should let bandit learn context-conditioned decisions`() {
        val schema = ContextConditionedSchema()
        val space = schema.compileSpace()
        val projection = LinearFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)
        val params = LocalSearchParams(maxFlips = 1_000L, randomSeed = 4L)
        val bandit = diagonalBandit(projection, solver, params, exploration = 0.0, randomSeed = 4)

        fun groundTruth(s: combo.decisions.BanditSample, premium: Boolean, segment: Int): Double {
            val c1 = if (s.bools[0]) 1.0 else 0.0
            val c2 = if (s.bools[1]) 1.0 else 0.0
            return (if (premium) 1.0 else -1.0) * c1 + (if (segment > 0) 1.0 else -1.0) * c2
        }

        val rng = Random(4)
        val contexts = listOf(
            Triple(true, 1, context { set(schema.premium, true); set(schema.segment, 1) }),
            Triple(true, -1, context { set(schema.premium, true); set(schema.segment, -1) }),
            Triple(false, 1, context { set(schema.premium, false); set(schema.segment, 1) }),
            Triple(false, -1, context { set(schema.premium, false); set(schema.segment, -1) }),
        )
        repeat(2000) {
            val (premium, segment, ctx) = contexts.random(rng)
            val asm = projection.assumptionsFor(ctx)
            val s = combo.decisions.BanditSample.undithered(solver.sample(params.copy(randomSeed = rng.nextLong(), assumptions = asm))!!)
            bandit.train(s, ctx, groundTruth(s, premium, segment))
        }

        var matches = 0
        for ((premium, segment, ctx) in contexts) {
            val best = bandit.optimalOrThrow(ctx)
            if (best.bools[0] == premium) matches++
            if (best.bools[1] == (segment > 0)) matches++
        }
        assertTrue(matches >= 6, "context-conditioned matches: $matches/8 (expected ≥ 6)")
    }

    @Test
    fun `inactive optional slots should be zeroed in encoding`() {
        val model = WithOptional()
        val space = model.compileSpace()
        val projection = LinearFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)
        val params = LocalSearchParams(maxFlips = 1_000L, randomSeed = 21L)

        val rng = Random(21)
        var inactiveSample: Sample? = null
        var activeSample: Sample? = null
        repeat(80) {
            val s = solver.sample(params.copy(randomSeed = rng.nextLong()))!!
            val gate = space.gateOf(model.audio)!!
            if (!space.compiled.decode(gate, s) && inactiveSample == null) inactiveSample = s
            if (space.compiled.decode(gate, s) && activeSample == null) activeSample = s
        }
        assertTrue(inactiveSample != null && activeSample != null,
            "solver should produce both gate-off and gate-on samples")

        val inactiveFeatures = projection.encode(combo.decisions.BanditSample.undithered(inactiveSample!!))
        val muteSlot = projection.layout.boolStart +
            space.compiled.boolVarIdByName["audio.mute"]!!
        assertEquals(0.0, inactiveFeatures[muteSlot])

        val activeFeatures = projection.encode(combo.decisions.BanditSample.undithered(activeSample!!))
        val expected = if (activeSample!!.bools[space.compiled.boolVarIdByName["audio.mute"]!!]) 1.0 else 0.0
        assertEquals(expected, activeFeatures[muteSlot])
    }

    @Test
    fun `missing context value should throw`() {
        val schema = TogglesWithContext()
        val space = schema.compileSpace()
        val projection = LinearFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)
        val bandit = diagonalBandit(
            projection, solver,
            LocalSearchParams(randomSeed = 1L),
            exploration = 1.0,
            priorPrecision = 1.0,
            randomSeed = 1,
        )

        val incompleteCtx = context {
            set(schema.premium, true)
            // segment intentionally missing
        }
        val ex = runCatching { bandit.chooseOrThrow(incompleteCtx) }.exceptionOrNull()
        assertTrue(ex != null, "missing context value should fail fast")
        assertTrue(ex!!.message!!.contains("segment"),
            "error should name the missing handle: got '${ex.message}'")
    }
}

private fun Random.nextGaussian(): Double {
    var u: Double; var s: Double
    do {
        u = nextDouble() * 2 - 1
        val v = nextDouble() * 2 - 1
        s = u * u + v * v
    } while (s >= 1.0 || s == 0.0)
    return u * sqrt(-2.0 * ln(s) / s)
}
