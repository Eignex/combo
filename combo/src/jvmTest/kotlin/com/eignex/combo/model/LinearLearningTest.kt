package com.eignex.combo.model

import com.eignex.combo.bandit.glm.LinearTrialOptimizer
import com.eignex.combo.decisions.DecisionSpaceDef
import com.eignex.combo.decisions.compileSpace
import com.eignex.combo.decisions.context
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.combo.shared.LinearFeatureProjection
import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.backtrack.BacktrackSolver
import com.eignex.klause.model.IntSpec
import com.eignex.klause.propagation.bake
import com.eignex.kumulant.schema.runtime.materialize
import com.eignex.kumulant.schema.spec.BayesianRegression
import com.eignex.kumulant.stat.regression.glm.MultivariateGaussian
import com.eignex.skema.SchemaJson
import kotlinx.serialization.encodeToString
import kotlin.test.*

class LinearLearningTest {
    @Test
    fun `native model and posterior retain canonical serialization`() {
        val text = """{"regression":{"bayesianRegression":{"featureSize":2}},"posterior":"multivariateGaussian"}"""
        val learning = SchemaJson.decodeFromString<LinearLearning>(text)
        assertEquals(BayesianRegression(2), learning.regression)
        assertEquals(learning, SchemaJson.decodeFromString<LinearLearning>(SchemaJson.encodeToString(learning)))
        assertTrue("bayesianRegression" in SchemaJson.encodeToString(learning))
        assertFailsWith<IllegalArgumentException> {
            SchemaJson.decodeFromString<LinearLearning>(text.replace("multivariateGaussian", "factorisedGaussian"))
        }
    }

    @Test
    fun `out of order completion trains native regression on the proposal original context`() {
        val space = DecisionSpaceDef(
            variables = mapOf("workers" to IntSpec(1, 8)),
            context = mapOf("load" to IntSpec(1, 8)),
        ).compileSpace()
        val projection = LinearFeatureProjection(space)
        val spec = BayesianRegression(projection.featureSize)
        val oracle = spec.materialize()
        val search = HybridDescentSession.klause(space)
        BacktrackSolver(space.compiled.problem.bake()).session().use { session ->
            val optimizer = LinearTrialOptimizer(
                projection, LinearLearning(spec, MultivariateGaussian), search,
                { objective, pins -> session.minimize(objective, BacktrackParams(assumptions = pins)).assignment },
            )
            val handle = space.contextInts.single()
            val firstContext = context { set(handle, 2) }
            val secondContext = context { set(handle, 6) }
            val first = optimizer.suggest(1, firstContext).single()
            val second = optimizer.suggest(1, secondContext).single()
            optimizer.complete(second.id, 90.0)
            oracle.update(projection.encode(second.sample, second.context), 90.0)
            optimizer.complete(first.id, 10.0)
            oracle.update(projection.encode(first.sample, first.context), 10.0)
            assertEquals(2L, optimizer.observationCount)
            for (trial in optimizer.trials) {
                val features = projection.encode(trial.sample, trial.context)
                assertEquals(oracle.read().predict(features), optimizer.predict(trial.sample, trial.context), 1e-10)
            }
            assertFailsWith<IllegalArgumentException> { optimizer.complete(first.id, 20.0) }
            assertEquals(2L, optimizer.observationCount)
            val pending = optimizer.suggest(1, secondContext).single()
            assertFailsWith<IllegalArgumentException> { optimizer.complete(pending.id, 5.0, noiseVariance = 1.0) }
            optimizer.cancel(pending.id)
            assertEquals(2L, optimizer.observationCount)
        }
    }
}
