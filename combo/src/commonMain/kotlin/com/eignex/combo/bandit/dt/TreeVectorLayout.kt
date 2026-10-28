package com.eignex.combo.bandit.dt

import com.eignex.klause.util.toDouble
import com.eignex.koblas.Vector
import com.eignex.kumulant.stat.regression.tree.SerializableSplit
import com.eignex.kumulant.stat.regression.tree.ThresholdSplit
import com.eignex.kumulant.stat.regression.tree.Split as VectorSplit

/**
 * Numeric view of typed tree inputs, retaining the Klause predicate for each split.
 * Numeric columns use declared values. Other predicates map true to zero and false
 * to one so Kumulant's `<= 0.5` split preserves the original positive branch.
 */
class TreeVectorLayout(candidates: List<Split>) {
    private val columns = mutableListOf<(TreeRow) -> Double>()
    private val columnIds = mutableMapOf<Any, Int>()
    private val predicates = mutableMapOf<SerializableSplit, Split>()

    /** Kumulant predicates over the numeric view, in the caller's candidate order. */
    val splits: List<SerializableSplit> = candidates.map { split ->
        val (key, threshold, read) = when (split) {
            is IntThresholdSplit -> if (split.threshold >= 9007199254740992L || split.threshold <= -9007199254740992L) {
                Triple(split.expr, 0.5, { row: TreeRow -> if (split.direction(row)) 0.0 else 1.0 })
            } else {
                Triple(
                    "int:${split.handle.name}",
                    split.threshold.toDouble(),
                    { row: TreeRow ->
                        if (row.space.coordinates.schema == null) {
                            row.sample.sample.exactIntValue(
                                row.space.coordinates.intVarIdByName.getValue(split.handle.name),
                            ).toDouble()
                        } else {
                            row.int(split.handle).toDouble()
                        }
                    },
                )
            }

            is FloatThresholdSplit -> Triple(
                "float:${split.handle.name}",
                split.threshold,
                { row: TreeRow -> row.float(split.handle) },
            )

            else -> Triple(split.expr, 0.5, { row: TreeRow -> if (split.direction(row)) 0.0 else 1.0 })
        }
        val id = columnIds.getOrPut(key) {
            columns.add(read)
            columns.lastIndex
        }
        ThresholdSplit(id, threshold).also { predicates[it] = split }
    }

    /** Empty candidate pools use one inert coordinate to satisfy regression arity. */
    val featureSize: Int get() = columns.size.coerceAtLeast(1)

    /** Read-only view; numeric values retain their declared units without quantization. */
    fun encode(row: TreeRow): Vector = object : Vector {
        override val size: Int get() = featureSize
        override fun get(i: Int): Double {
            if (i !in 0 until size) throw IndexOutOfBoundsException("feature index $i outside 0 until $size")
            return if (columns.isEmpty()) 0.0 else columns[i](row)
        }
        override fun toDoubleArray(): DoubleArray = DoubleArray(size, ::get)
    }

    internal fun matches(candidates: List<Split>): Boolean =
        candidates.map { it.expr } == splits.map { predicate(it).expr }

    /** Recover the typed predicate for constraint-aware descent. */
    fun predicate(split: VectorSplit<Vector>): Split =
        predicates[split] ?: error("tree split is not part of this feature layout: $split")
}
