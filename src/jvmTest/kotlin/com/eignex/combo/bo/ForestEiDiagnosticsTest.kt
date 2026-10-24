package com.eignex.combo.bo

import com.eignex.combo.decisions.DecisionSpace
import com.eignex.combo.shared.HybridDescentSession
import com.eignex.kumulant.stat.regression.tree.BatchForestConfig
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals

/** Ablate replication and plateau ordering without asserting a stochastic performance ranking. */
class ForestEiDiagnosticsTest {
    private class Quadratic : DecisionSpace() {
        val x by floatVar(0.0, 1.0)
    }
    private class SearchTrace {
        var maximum = 0.0
        var ties = 0
        var scored = 0
    }

    private fun trace(group: String, target: Double, variant: String, seed: Int): List<String> {
        val model = Quadratic()
        val space = model.compileSpace()
        val projection = SearchTransform(space)
        val delegate = LocalAndRandomSearch(
            projection,
            HybridDescentSession.klause(space, candidateBatchSize = 32, poolCapacity = 128),
            candidateCount = 32,
            maxAttempts = 64,
            localSteps = 6,
            neighbors = 12,
            tieBreak = if (variant in listOf("ties", "both")) AcquisitionTieBreak.Random else AcquisitionTieBreak.First,
        )
        val trace = SearchTrace()
        val search = AcquisitionOptimizer { acquisition, request, random ->
            val scores = mutableListOf<Double>()
            val selected = delegate.maximize(
                Acquisition { features ->
                    acquisition.evaluate(features).also { scores += it.toDoubleArray().toList() }
                },
                request,
                random,
            )
            trace.maximum = scores.maxOrNull() ?: 0.0
            trace.ties = scores.count { it == trace.maximum }
            trace.scored = scores.size
            selected
        }
        val optimizer = BayesianOptimizer(
            projection,
            ForestSurrogate(BatchForestConfig(nbrTrees = 8)),
            acquisitionOptimizer = search,
            randomSeed = seed,
            repeatPolicy = if (variant in listOf("novel", "both")) RepeatPolicy.Never else RepeatPolicy.Allow,
        )
        val runner: TrialOptimizer = if (variant == "random") {
            StudyRunner(
                Study(space),
                RandomSuggestionPolicy(
                    space,
                    HybridDescentSession.klause(space, candidateBatchSize = 32, poolCapacity = 128),
                    seed,
                ),
            )
        } else {
            optimizer
        }
        val rows = mutableListOf<String>()
        val seen = mutableSetOf<Double>()
        repeat(32) { index ->
            val trial = runner.suggest(1)[0]
            val x = trial.sample.float(model.x, space)
            val repeated = !seen.add(x)
            val reward = -(x - target) * (x - target)
            val modelDriven = optimizer.lastSuggestion?.modelFits?.let { it > 0 } ?: false
            rows += "$group,$target,$variant,$seed,${index + 1},$x,$reward,$repeated,$modelDriven," +
                "${trace.maximum},${trace.ties},${trace.scored},${seen.size}"
            runner.complete(trial.id, reward)
        }
        assertEquals(32, runner.trials.count { it.status == TrialStatus.Completed })
        if (variant in listOf("novel", "both")) assertEquals(32, seen.size)
        return rows
    }

    @Test
    fun `trace quadratic evaluations and acquisition plateaus across discovery and validation cases`() {
        val rows = mutableListOf(
            "group,target,variant,seed,step,x,reward,repeated,model_driven," +
                "max_scored_acquisition,tied_maximum,scored,unique_so_far",
        )
        val groups = mapOf(
            "discovery" to listOf(0.73).flatMap { target -> listOf(3, 11, 19, 29, 43, 61).map { target to it } },
            "validation" to listOf(
                0.17,
                0.43,
                0.86,
            ).flatMap { target -> listOf(2, 5, 7, 13, 23, 37).map { target to it } },
        )
        for ((group, cases) in groups) {
            for ((target, seed) in cases) {
                for (variant in listOf("baseline", "novel", "ties", "both", "random")) {
                    rows += trace(group, target, variant, seed)
                }
            }
        }
        val directory = File("build/reports/bo-diagnostics").also { it.mkdirs() }
        File(directory, "quadratic.csv").writeText(rows.joinToString("\n", postfix = "\n"))
    }
}
