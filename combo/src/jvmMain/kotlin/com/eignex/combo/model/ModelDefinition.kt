package com.eignex.combo.model

import com.eignex.combo.decisions.DecisionSpaceDef
import com.eignex.klause.model.BoolSpec
import com.eignex.klause.model.FloatSpec
import com.eignex.klause.model.IntSpec
import com.eignex.klause.model.VarSpec
import com.eignex.kumulant.schema.spec.StatSpec
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
    /** Named variable or measured outcome to optimize; absent means feasibility only. */
    val objective: ModelObjective? = null,
    val outcome: String? = null,
    val stat: JsonObject? = null,
    /** Seed for reproducible solver choices and experimental proposals. */
    val seed: Int = 0,
    /** Named, native Kumulant specifications sharing this model's lifetime. */
    val statistics: Map<String, StatSpec> = emptyMap(),
    /** Statistical input names used by native Klause references. */
    val inputs: Map<String, StatisticalInput> = emptyMap(),
) {
    /** Whether the objective is measured through proposal feedback. */
    val learned: Boolean get() = outcome != null

    init {
        require(space != null || stat != null || statistics.isNotEmpty()) { "declare a decision space or a statistic" }
        require(inputs.isEmpty() || (space != null && outcome == null)) {
            "statistical inputs currently require a decision space with a known objective"
        }
        require(statistics.keys.all { it.isNotBlank() }) { "statistic names must be nonempty" }
        for ((name, input) in inputs) {
            require(name.isNotBlank() && input.statistic in statistics) { "unknown statistic for input '$name'" }
            require(inputs.keys.none { it != name && (it.startsWith("$name.") || name.startsWith("$it.")) }) {
                "statistical input names cannot overlap"
            }
        }
        require(space != null || (objective == null && outcome == null)) {
            "an objective or outcome requires a decision space"
        }
        val variables = space?.toSchemaDef()?.entries?.filterValues { it is VarSpec }.orEmpty()
        require(variables.keys.none { name -> inputs.keys.any { name == it || name.startsWith("$it.") } }) {
            "statistical inputs cannot shadow decision or context variables"
        }
        if (space != null) {
            val hasDecision = variables.keys.any { it !in space.context && it !in space.optionalContext }
            require(inputs.isNotEmpty() || hasDecision) { "declare a decision variable" }
        }
        if (outcome != null) {
            require(outcome.isNotBlank() && outcome !in variables) { "outcome must name a separate measured value" }
            require(objective?.name == outcome) { "objective must name the measured outcome" }
        } else {
            objective?.let {
                val spec = variables[it.name]
                val input = inputs.entries.firstOrNull { (name, _) -> it.name == name || it.name.startsWith("$name.") }
                require(spec == BoolSpec || spec is IntSpec || spec is FloatSpec || input != null) {
                    "objective must name a numeric or Boolean variable"
                }
                require(input == null || it.name != input.key || input.value.value != null) {
                    "select an explicit statistical projection for the objective"
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
data class ModelObjective(
    /** Name to minimize; mutually exclusive with [maximize]. */
    val minimize: String? = null,
    /** Name to maximize; mutually exclusive with [minimize]. */
    val maximize: String? = null,
) {
    /** Name selected by this objective. */
    val name: String get() = minimize ?: maximize!!
    /** Whether smaller objective values are preferred. */
    val ascending: Boolean get() = minimize != null

    init {
        require((minimize == null) != (maximize == null)) { "specify minimize or maximize" }
    }
}

/** Inputs to evaluating, recommending, or proposing decisions. */
@Serializable
data class ModelQuery(
    /** Named context values held fixed for this request. */
    val given: JsonObject = JsonObject(emptyMap()),
    /** Complete decision assignment when evaluating a supplied configuration. */
    val values: JsonObject = JsonObject(emptyMap()),
    /** Requested proposal batch size, from 1 to 64. */
    val count: Int = 1,
    /** Operation deadline including admission waiting, from 1 to 300,000 milliseconds. */
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
    /** Identifier of the pending proposal whose original assignment is retained. */
    val proposal: Long,
    /** Measured values keyed by outcome name. */
    val values: JsonObject,
    /** Positive finite observation weight. */
    val weight: Double = 1.0,
    /** Optional nonnegative finite measurement variance in outcome units squared. */
    val noiseVariance: Double? = null,
) {
    init {
        require(weight.isFinite() && weight > 0) { "weight must be positive and finite" }
        require(noiseVariance == null || (noiseVariance.isFinite() && noiseVariance >= 0)) {
            "noiseVariance must be nonnegative and finite"
        }
    }
}

/** Update one named statistic using its unchanged native observation fields. */
@Serializable
data class StatisticalObservation(
    /** Statistic declared in [ModelDefinition.statistics]. */
    val statistic: String,
    /** Native scalar, discrete, paired, vector, or regression observation. */
    val observation: JsonObject,
)
