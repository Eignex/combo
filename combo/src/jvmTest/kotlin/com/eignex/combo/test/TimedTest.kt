package com.eignex.combo.test

import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.decisions.minimizeNative
import com.eignex.combo.model.ModelDefinition
import com.eignex.combo.model.ModelStatistics
import com.eignex.klause.backtrack.BacktrackParams
import com.eignex.klause.backtrack.BacktrackSolver
import com.eignex.klause.formats.json.ProblemJson
import com.eignex.klause.propagation.bake
import com.eignex.koblas.DenseVector
import com.eignex.koblas.dot
import com.eignex.skema.SchemaJson
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import org.junit.BeforeClass
import org.junit.Rule
import org.junit.rules.Timeout

actual abstract class TimedTest actual constructor() {
    @get:Rule
    val timeout: Timeout = Timeout.millis(300)

    companion object {
        private class RuntimeProbe : DecisionSpace() {
            val flag by boolVar()
            val count by intVar(0, 1)
            val rate by floatVar(0.0, 1.0)
        }

        // Load the compiler, codecs, solver and SIMD backend before timing individual checks.
        private val runtime by lazy {
            val vector = DenseVector.of(doubleArrayOf(1.0))
            check((vector dot vector) == 1.0)
            val model = RuntimeProbe()
            val space = model.compileSpace()
            SchemaJson.encodeToString(space.definition)
            val definition = SchemaJson.decodeFromString<ModelDefinition>(
                """{"space":{"variables":{"flag":"bool"}},"statistics":{"probe":{"covariance":{}}}}""",
            )
            ModelStatistics(definition).use { statistics ->
                statistics.observe(
                    "probe",
                    buildJsonObject {
                        put("x", 0)
                        put("y", 0)
                    },
                )
                statistics.snapshot()
            }
            val solver = BacktrackSolver(space.compiled.problem.bake())
            solver.sample(BacktrackParams(maxDecisions = 8))
            val native = CompiledDecisionSpace.fromProblem(
                ProblemJson.default.decodeFromString(
                    """{"ints":[{"lower":0,"upper":1}]}""",
                ),
            )
            native.minimizeNative(native.coordinates.objective("ints.0"), Context.Empty)
        }

        @BeforeClass
        @JvmStatic
        fun initializeRuntime() {
            runtime
        }
    }
}
