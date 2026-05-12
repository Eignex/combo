package combo.bandit.glm

import com.eignex.klause.solver.LocalSearchParams
import com.eignex.klause.solver.LocalSearchSolver
import com.eignex.klause.solver.Sample
import combo.decisions.DecisionSpace
import combo.decisions.SubSpace
import combo.decisions.context
import kotlin.math.abs
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
    val segment by contextInt()
}

private class AudioBlock : SubSpace() {
    val mute by boolVar()
}

private class WithOptional : DecisionSpace() {
    val baseline by boolVar()
    val audio by optionalSubmodel(::AudioBlock)
}

private class WithNominal : DecisionSpace() {
    val type by nominal("a", "b", "c")
}

private class ContextConditionedSchema : DecisionSpace() {
    val choice1 by boolVar()
    val choice2 by boolVar()
    val premium by contextBool()
    val segment by contextInt()
    val premium_x_c1 by interact(premium, choice1)
    val segment_x_c2 by interact(segment, choice2)
}

class LinearBanditTest {

    @Test
    fun thompsonSamplingConvergesToBestSampleNoContext() {
        val space = FiveToggles().compileSpace()
        val projection = LinearFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)
        val params = LocalSearchParams(maxFlips = 1_000L, randomSeed = 11L)

        val model = DiagonalizedLinearModel.Builder(projection.featureSize)
            .family(NormalVariance)
            .learningRate(ConstantRate(1f))
            .priorPrecision(0.01f)
            .exploration(0.3f)
            .build()
        val bandit = LinearBandit(
            projection = projection,
            linearModel = model,
            innerOptimizer = { obj -> solver.minimize(obj, params) },
            randomSeed = 31,
        )

        // Reward = Σ w_i * bool_i (raw 0/1 features). Best arms: a, c, e set; b, d unset.
        val trueWeights = doubleArrayOf(+1.0, -1.0, +1.0, -1.0, +1.0)
        fun groundTruth(sample: Sample): Double {
            var r = 0.0
            for (i in 0 until 5) r += trueWeights[i] * (if (sample.bools[i]) 1.0 else 0.0)
            return r
        }
        val optimalReward = trueWeights.filter { it > 0 }.sum()  // = 3.0

        val rng = Random(31)
        var cumulativeRegret = 0.0
        val rounds = 1500
        repeat(rounds) {
            val sample = bandit.chooseOrThrow()
            val reward = groundTruth(sample) + rng.nextGaussian() * 0.05
            cumulativeRegret += optimalReward - groundTruth(sample)
            bandit.update(sample, reward)
        }
        assertTrue(cumulativeRegret / rounds < 0.5,
            "Average regret too high: ${cumulativeRegret / rounds} over $rounds rounds")

        val best = bandit.optimalOrThrow()
        for (i in 0 until 5) {
            assertEquals(trueWeights[i] > 0, best.bools[i], "bool $i mismatch in optimal sample")
        }
    }

    @Test
    fun fixedContextSteersBandit() {
        val schema = TogglesWithContext()
        val space = schema.compileSpace()
        val projection = LinearFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)
        val params = LocalSearchParams(maxFlips = 1_000L, randomSeed = 17L)

        val model = DiagonalizedLinearModel.Builder(projection.featureSize)
            .family(NormalVariance)
            .learningRate(ConstantRate(1f))
            .priorPrecision(0.01f)
            .exploration(0f)
            .build()
        val bandit = LinearBandit(
            projection = projection,
            linearModel = model,
            innerOptimizer = { obj -> solver.minimize(obj, params) },
            randomSeed = 17,
        )

        // Reward = +choice1 + choice2 (always prefers both on). With a fixed context
        // and a model that has additive context features, the bandit should learn the
        // optimal decision under that fixed context.
        val ctx = context {
            set(schema.premium, true)
            set(schema.segment, 1)
        }
        fun groundTruth(s: Sample): Double {
            return (if (s.bools[0]) 1.0 else 0.0) + (if (s.bools[1]) 1.0 else 0.0)
        }

        val rng = Random(17)
        repeat(800) {
            val s = solver.sample(params.copy(randomSeed = rng.nextLong()))!!
            bandit.train(s, ctx, groundTruth(s))
        }

        val best = bandit.optimalOrThrow(ctx)
        assertTrue(best.bools[0], "choice1 should be picked under positive-reward training")
        assertTrue(best.bools[1], "choice2 should be picked under positive-reward training")
    }

    @Test
    fun nominalDecisionsExpandToPerLabelFeatures() {
        val schema = WithNominal()
        val space = schema.compileSpace()
        val projection = LinearFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)
        val params = LocalSearchParams(maxFlips = 1_000L, randomSeed = 123L)

        // klause expands the 3-label nominal into 3 indicator bools — one feature per label.
        assertEquals(3, space.compiled.problem.numBoolVars)
        assertEquals(3, projection.featureSize)
        val indicators = space.compiled.nominalIndicators[schema.type.name]!!
        assertEquals(setOf("a", "b", "c"), indicators.keys)

        // Linear bandit learns label-specific weights. Reward favours label "b".
        val labelReward = mapOf("a" to 0.0, "b" to 1.0, "c" to 0.2)
        val model = DiagonalizedLinearModel.Builder(projection.featureSize)
            .family(NormalVariance)
            .learningRate(ConstantRate(1f))
            .priorPrecision(0.01f)
            .exploration(0.1f)
            .build()
        val bandit = LinearBandit(
            projection = projection,
            linearModel = model,
            innerOptimizer = { obj -> solver.minimize(obj, params) },
            randomSeed = 123,
        )
        val rng = Random(123)
        repeat(800) {
            val sample = bandit.chooseOrThrow()
            val picked = space.compiled.decode(schema.type, sample)
            val noisy = labelReward[picked]!! + rng.nextGaussian() * 0.05
            bandit.update(sample, noisy)
        }
        val best = bandit.optimalOrThrow()
        assertEquals("b", space.compiled.decode(schema.type, best))
    }

    @Test
    fun interactionsLetBanditLearnContextConditionedDecisions() {
        val schema = ContextConditionedSchema()
        val space = schema.compileSpace()
        val projection = LinearFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)
        val params = LocalSearchParams(maxFlips = 1_000L, randomSeed = 4L)

        val model = DiagonalizedLinearModel.Builder(projection.featureSize)
            .family(NormalVariance)
            .learningRate(ConstantRate(1f))
            .priorPrecision(0.01f)
            .exploration(0f)
            .build()
        val bandit = LinearBandit(
            projection = projection,
            linearModel = model,
            innerOptimizer = { obj -> solver.minimize(obj, params) },
            randomSeed = 4,
        )

        // Reward depends on context × decision: positive premium + segment flip
        // the best decision for each choice.
        //   reward = (premium ? +1 : -1) * choice1 + (segment > 0 ? +1 : -1) * choice2
        fun groundTruth(s: Sample, premium: Boolean, segment: Int): Double {
            val c1 = if (s.bools[0]) 1.0 else 0.0
            val c2 = if (s.bools[1]) 1.0 else 0.0
            return (if (premium) 1.0 else -1.0) * c1 + (if (segment > 0) 1.0 else -1.0) * c2
        }

        val rng = Random(4)
        // Train across all four context combinations.
        val contexts = listOf(
            Triple(true, 1, context { set(schema.premium, true); set(schema.segment, 1) }),
            Triple(true, -1, context { set(schema.premium, true); set(schema.segment, -1) }),
            Triple(false, 1, context { set(schema.premium, false); set(schema.segment, 1) }),
            Triple(false, -1, context { set(schema.premium, false); set(schema.segment, -1) }),
        )
        repeat(2000) {
            val (premium, segment, ctx) = contexts.random(rng)
            val s = solver.sample(params.copy(randomSeed = rng.nextLong()))!!
            bandit.train(s, ctx, groundTruth(s, premium, segment))
        }

        // For each context, the bandit's optimal sample matches the per-context optimum.
        for ((premium, segment, ctx) in contexts) {
            val best = bandit.optimalOrThrow(ctx)
            val c1Optimal = premium
            val c2Optimal = segment > 0
            assertEquals(c1Optimal, best.bools[0],
                "context (premium=$premium, segment=$segment): choice1 should be $c1Optimal")
            assertEquals(c2Optimal, best.bools[1],
                "context (premium=$premium, segment=$segment): choice2 should be $c2Optimal")
        }
    }

    @Test
    fun inactiveOptionalSlotsAreZeroedInEncoding() {
        val model = WithOptional()
        val space = model.compileSpace()
        val projection = LinearFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)
        val params = LocalSearchParams(maxFlips = 1_000L, randomSeed = 21L)

        // Find a sample with the audio gate OFF (the auto-allocated bool named "audio"
        // is at id 1, since baseline is at id 0). Pinning forces audio.mute = false.
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

        // Inactive: the audio.mute slot must be 0 in the feature vector regardless of
        // its (pinned-to-false) klause value.
        val inactiveFeatures = projection.encode(inactiveSample!!)
        val muteSlot = projection.layout.boolDecisionsStart +
            space.compiled.boolVarIdByName["audio.mute"]!!
        assertEquals(0f, inactiveFeatures[muteSlot])

        // Active: the audio.mute slot reflects the actual sampled value.
        val activeFeatures = projection.encode(activeSample!!)
        val expected = if (activeSample!!.bools[space.compiled.boolVarIdByName["audio.mute"]!!]) 1f else 0f
        assertEquals(expected, activeFeatures[muteSlot])
    }

    @Test
    fun missingContextValueThrows() {
        val schema = TogglesWithContext()
        val space = schema.compileSpace()
        val projection = LinearFeatureProjection(space)
        val solver = LocalSearchSolver(space.compiled.problem)

        val model = DiagonalizedLinearModel.Builder(projection.featureSize).build()
        val bandit = LinearBandit(
            projection = projection,
            linearModel = model,
            innerOptimizer = { obj -> solver.minimize(obj, LocalSearchParams(randomSeed = 1L)) },
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
