package com.eignex.combo.bo

import com.eignex.kumulant.stat.score.MseLossStat
import com.eignex.kumulant.stat.summary.MeanStat
import java.io.File
import kotlin.math.abs
import kotlin.math.sqrt

private data class FrozenPrediction(val mean: Double, val variance: Double, val truth: Double)

internal fun scoreSmacPredictions(input: File, output: File) {
    output.printWriter().use { writer ->
        writer.println("problem,seed,history,uncertainty,rmse,coverage_two_sd,mean_sd,top_rmse,top_coverage_two_sd")
        var group: List<String>? = null
        val completed = mutableSetOf<List<String>>()
        val rows = mutableListOf<FrozenPrediction>()
        fun flush() {
            if (rows.isEmpty()) return
            val loss = MseLossStat()
            val topLoss = MseLossStat()
            val coverage = MeanStat()
            val topCoverage = MeanStat()
            val width = MeanStat()
            val threshold = rows.map { it.truth }.sorted()[rows.size / 10]
            for (row in rows) {
                val deviation = sqrt(row.variance)
                val covered = if (abs(row.mean - row.truth) <= 2.0 * deviation) 1.0 else 0.0
                loss.update(row.mean, row.truth, 0L, 1.0)
                coverage.update(covered, 0L, 1.0)
                width.update(deviation, 0L, 1.0)
                if (row.truth <= threshold) {
                    topLoss.update(row.mean, row.truth, 0L, 1.0)
                    topCoverage.update(covered, 0L, 1.0)
                }
            }
            writer.println(
                requireNotNull(group).joinToString(",") + ",smac-2.4.1,${sqrt(loss.read(0L).mean)}," +
                    "${coverage.read(0L).mean},${width.read(0L).mean},${sqrt(topLoss.read(0L).mean)}," +
                    topCoverage.read(0L).mean,
            )
            completed += requireNotNull(group)
            rows.clear()
        }
        input.useLines { lines ->
            val iterator = lines.iterator()
            require(iterator.hasNext() && iterator.next() == "problem,seed,history,prediction,variance,truth") {
                "expected raw SMAC prediction columns"
            }
            for (line in iterator) {
                val columns = line.split(',')
                require(columns.size == 6)
                val key = columns.take(3)
                if (key != group) {
                    flush()
                    require(key !in completed) { "SMAC prediction groups must be contiguous" }
                    group = key
                }
                val row = FrozenPrediction(columns[3].toDouble(), columns[4].toDouble(), columns[5].toDouble())
                require(row.mean.isFinite() && row.truth.isFinite() && row.variance >= 0.0 && row.variance.isFinite())
                rows += row
            }
            require(rows.isNotEmpty()) { "expected SMAC prediction rows" }
            flush()
        }
    }
}

/** Score exported reference predictions with Kumulant using the refinement report schema. */
fun main() {
    val root = File("build/reports/bo-refinement")
    scoreSmacPredictions(File(root, "smac-predictions.csv"), File(root, "smac-models.csv"))
}
