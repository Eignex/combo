package com.eignex.combo.bo

import com.eignex.kumulant.stat.score.MaeLossStat
import com.eignex.kumulant.stat.summary.MeanStat
import java.io.File
import kotlin.math.exp

/** Score archived acquisition estimates using Kumulant losses; independent references remain explicit. */
internal fun scoreGpConvergence(directory: File) {
    val rows = csvRows(File(directory, "convergence.csv"))
    val sobol = File(directory, "sobol.csv").takeIf { it.exists() }?.let(::csvRows) ?: emptyList()
    val output = File(directory, "convergence-scores.csv")
    output.writeText(
        "fixture,q,reference,kind,replicate,samples,mae,reference_best,selected_reference," +
            "relative_regret,exact_match,within_one_percent,zero_fraction\n",
    )
    for ((key, fixture) in rows.groupBy { it.getValue("fixture") to it.getValue("q") }) {
        val references = listOf("iid-8", "iid-9", "sobol-0", "sobol-1", "sobol-2")
        for (reference in references) {
            val precise = if (reference.startsWith("sobol-")) {
                sobol.filter {
                    it.getValue("fixture") == key.first && it.getValue("q") == key.second &&
                        it.getValue("replicate") == reference.removePrefix("sobol-")
                }
            } else {
                fixture.filter { it.getValue("replicate") == reference.removePrefix("iid-") }
            }.associateBy { it.getValue("set") }
            if (precise.isEmpty()) continue
            for ((run, estimates) in fixture.groupBy { it.getValue("replicate") to it.getValue("samples") }) {
                for (kind in listOf("raw", "log")) {
                    val loss = MaeLossStat()
                    val zero = MeanStat()
                    for (row in estimates) {
                        val estimate = row.getValue(kind).toDouble()
                        loss.update(estimate, precise.getValue(row.getValue("set")).getValue(kind).toDouble(), 0L)
                        zero.update(if (estimate == 0.0 && kind == "raw") 1.0 else 0.0, 0L)
                    }
                    val selected = estimates.maxBy { it.getValue(kind).toDouble() }.getValue("set")
                    val best = precise.values.maxOf { it.getValue(kind).toDouble() }
                    val actual = precise.getValue(selected).getValue(kind).toDouble()
                    val regret = if (kind == "log") {
                        1 - exp(actual - best)
                    } else if (best > 0) {
                        1 - actual / best
                    } else {
                        0.0
                    }
                    val exact = if (actual == best) 1 else 0
                    val close = if (regret <= 0.01) 1 else 0
                    output.appendText(
                        "${key.first},${key.second},$reference,$kind,${run.first},${run.second}," +
                            "${loss.read(0L).mean},$best,$actual,$regret,$exact,$close,${zero.read(0L).mean}\n",
                    )
                }
            }
        }
    }
}

private fun csvRows(file: File): List<Map<String, String>> = file.readLines().let { lines ->
    val columns = lines.first().split(',')
    lines.drop(1).map { columns.zip(it.split(',')).toMap() }
}
