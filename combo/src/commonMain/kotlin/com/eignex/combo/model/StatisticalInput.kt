package com.eignex.combo.model

import com.eignex.klause.model.IntExpr
import com.eignex.klause.model.IntRef
import com.eignex.klause.model.IntTerm
import com.eignex.klause.schema.FloatExpr
import com.eignex.klause.schema.FloatHandle
import com.eignex.klause.schema.FloatTerm
import kotlinx.serialization.Serializable

/** A view of one native statistic result; it introduces no expression nodes. */
@Serializable
data class StatisticalInput(
    /** Named statistic in [ModelDefinition.statistics]. */
    val statistic: String,
    /** Object/array path within that statistic's result, for example a shared group's child. */
    val path: List<String> = emptyList(),
    /** Optional projection for a fixed scalar input; absent means an interval with lower/upper outputs. */
    val value: String? = null,
    /** Optional aliases to native result paths; unaliased projections select same-named fields. */
    val projections: Map<String, List<String>> = emptyMap(),
) {
    init {
        require(statistic.isNotBlank()) { "statistic must be nonempty" }
        require(value == null || value.isNotBlank()) { "value projection must be nonempty" }
        require(projections.keys.all { it.isNotBlank() && '.' !in it }) {
            "projection names must be simple nonempty names"
        }
    }
}

/** Integer view of a statistical input, using Klause's existing integer DSL operators. */
class StatisticalInt(
    /** Input name registered in [ModelDefinition.inputs]. */
    val name: String,
) : IntTerm {
    /** Explicit lower bound from the shared snapshot. */
    val lower: IntExpr get() = projection("lower")

    /** Explicit upper bound from the shared snapshot. */
    val upper: IntExpr get() = projection("upper")

    /** Mean, if the statistic exposes it or the binding provides an alias. */
    val mean: IntExpr get() = projection("mean")

    /** Median, if the statistic exposes it or the binding provides an alias. */
    val median: IntExpr get() = projection("median")

    /** Select a named scalar output; integer use requires an exactly representable Int value. */
    fun projection(name: String): IntExpr = IntRef("${this.name}.$name")

    override fun toIntExpr(): IntExpr = IntRef(name)
}

/** Real view using Klause's existing linear float DSL and native constraint serializer. */
class StatisticalReal(
    /** Input name registered in [ModelDefinition.inputs]. */
    val name: String,
) : FloatTerm {
    /** Explicit lower bound from the shared snapshot. */
    val lower: FloatTerm get() = projection("lower")

    /** Explicit upper bound from the shared snapshot. */
    val upper: FloatTerm get() = projection("upper")

    /** Mean, if the statistic exposes it or the binding provides an alias. */
    val mean: FloatTerm get() = projection("mean")

    /** Median, if the statistic exposes it or the binding provides an alias. */
    val median: FloatTerm get() = projection("median")

    /** Select a native scalar output using an ordinary Klause handle. */
    fun projection(name: String): FloatTerm = FloatHandle("${this.name}.$name", -Double.MAX_VALUE, Double.MAX_VALUE)

    override fun toFloatExpr(): FloatExpr = FloatHandle(name, -Double.MAX_VALUE, Double.MAX_VALUE).toFloatExpr()
}
