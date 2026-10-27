package com.eignex.combo.model

import com.eignex.combo.decisions.DecisionSpaceDef
import com.eignex.klause.model.BoolSpec
import com.eignex.klause.model.FloatSpec
import com.eignex.klause.model.IntSpec
import com.eignex.klause.model.VarSpec
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject

/**
 * A decision model, a statistic, or a decision model learning a measured [outcome].
 * [space] retains the native variable and constraint vocabulary of [DecisionSpaceDef].
 * [stat] carries a canonical Kumulant statistic or bandit specification, decoded by its
 * native serializer when materialized. No expression or variable translation is involved.
 */
@Serializable
data class ModelDefinition(
    val space: DecisionSpaceDef? = null,
    val objective: ModelObjective? = null,
    val outcome: String? = null,
    val stat: JsonObject? = null,
    val seed: Int = 0,
) {
    /** Whether the objective is measured through proposal feedback. */
    val learned: Boolean get() = outcome != null

    init {
        require(space != null || stat != null) { "declare a decision space or a statistic" }
        require(space != null || (objective == null && outcome == null)) {
            "an objective or outcome requires a decision space"
        }
        val variables = space?.toSchemaDef()?.entries?.filterValues { it is VarSpec }.orEmpty()
        if (space != null) {
            require(variables.keys.any { it !in space.context && it !in space.optionalContext }) {
                "declare a decision variable"
            }
        }
        if (outcome != null) {
            require(outcome.isNotBlank() && outcome !in variables) { "outcome must name a separate measured value" }
            require(objective?.name == outcome) { "objective must name the measured outcome" }
        } else {
            objective?.let {
                val spec = variables[it.name]
                require(spec == BoolSpec || spec is IntSpec || spec is FloatSpec) {
                    "objective must name a numeric or Boolean variable"
                }
            }
        }
        require(stat == null || space == null || learned) {
            "an attached statistic requires a measured outcome"
        }
    }
}

/** Select one named numeric variable or measured outcome to minimize or maximize. */
@Serializable
data class ModelObjective(val minimize: String? = null, val maximize: String? = null) {
    val name: String get() = minimize ?: maximize!!
    val ascending: Boolean get() = minimize != null

    init {
        require((minimize == null) != (maximize == null)) { "specify minimize or maximize" }
    }
}

/** Inputs to evaluating, recommending, or proposing decisions. */
@Serializable
data class ModelQuery(
    val given: JsonObject = JsonObject(emptyMap()),
    val values: JsonObject = JsonObject(emptyMap()),
    val count: Int = 1,
    val timeoutMillis: Long = 30_000,
) {
    init {
        require(timeoutMillis in 1..300_000) { "timeoutMillis must be between 1 and 300000" }
        require(count in 1..64) { "count must be between 1 and 64" }
    }
}

/** Feedback for one retained proposal, using its original decision and context. */
@Serializable
data class ModelObservation(
    val proposal: Long,
    val values: JsonObject,
    val weight: Double = 1.0,
    val noiseVariance: Double? = null,
) {
    init {
        require(weight.isFinite() && weight > 0) { "weight must be positive and finite" }
        require(noiseVariance == null || (noiseVariance.isFinite() && noiseVariance >= 0)) {
            "noiseVariance must be nonnegative and finite"
        }
    }
}
