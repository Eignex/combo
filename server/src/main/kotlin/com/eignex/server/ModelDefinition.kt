package com.eignex.server

import com.eignex.combo.decisions.DecisionSpaceDef
import com.eignex.klause.model.*
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject

@Serializable
internal data class ModelDefinition(
    val variables: Map<String, ModelVariable> = emptyMap(),
    val constraints: List<ModelConstraint> = emptyList(),
    val objective: ModelObjective? = null,
    val stat: JsonObject? = null,
    val seed: Int = 0,
) {
    val learned: Boolean get() = objective?.name?.let { variables.getValue(it).role == "observed" } == true

    init {
        require(variables.isNotEmpty() || stat != null) { "declare variables or a statistic" }
        require(variables.keys.all { it.matches(Regex("[A-Za-z][A-Za-z0-9_]*")) }) { "invalid variable name" }
        variables.values.forEach { it.validate() }
        objective?.let {
            val variable = requireNotNull(variables[it.name]) { "unknown objective variable '${it.name}'" }
            require(variable.type != "categorical") { "the objective must be numeric or Boolean" }
        }
        val observed = variables.filterValues { it.role == "observed" }.keys
        require(observed.isEmpty() || observed == setOf(objective?.name)) {
            "observed variables must name the objective outcome"
        }
        require(stat == null || variables.isEmpty() || learned) {
            "a statistic can stand alone or track an observed objective"
        }
        require(variables.isEmpty() || variables.values.any { it.role == "decision" }) { "declare a decision variable" }
        require(variables.isNotEmpty() || (constraints.isEmpty() && objective == null)) {
            "constraints and objectives require variables"
        }
        constraints.forEach { it.expression(variables) }
    }

    fun space(): DecisionSpaceDef = DecisionSpaceDef(
        variables = variables.filterValues { it.role == "decision" }.mapValues { it.value.spec() },
        context = variables.filterValues { it.role == "context" }.mapValues { it.value.spec() },
        constraints = constraints.mapIndexed { index, constraint ->
            "_constraint_$index" to constraint.expression(variables)
        }.toMap(),
    )
}

@Serializable
internal data class ModelVariable(
    val type: String = "real",
    val role: String = "decision",
    val min: Double? = null,
    val max: Double? = null,
    val labels: List<String> = emptyList(),
) {
    fun validate() {
        require(role in setOf("decision", "context", "observed")) { "unknown variable role '$role'" }
        require(type in setOf("integer", "real", "boolean", "categorical")) { "unknown variable type '$type'" }
        if (role == "observed") {
            require(type == "real" && min == null && max == null && labels.isEmpty()) {
                "observed outcomes are real values without search bounds"
            }
        } else {
            if (role == "context") require(type in setOf("integer", "boolean")) {
                "context variables must be integer or Boolean"
            }
            when (type) {
                "integer", "real" -> {
                    require(min != null && max != null && min.isFinite() && max.isFinite() && min <= max) {
                        "numeric decisions and contexts require finite min and max bounds"
                    }
                    require(labels.isEmpty()) { "numeric variables cannot have labels" }
                    if (type == "integer") {
                        require(min.toInt().toDouble() == min && max.toInt().toDouble() == max) {
                            "integer bounds must be 32-bit integers"
                        }
                    }
                }
                "boolean" -> require(min == null && max == null && labels.isEmpty()) { "Boolean variables have no bounds" }
                "categorical" -> require(min == null && max == null && labels.isNotEmpty() && labels.distinct() == labels) {
                    "categorical variables require unique labels"
                }
            }
        }
    }

    fun spec(): VarSpec = when (type) {
        "integer" -> IntSpec(min!!.toInt(), max!!.toInt())
        "real" -> FloatSpec(min!!, max!!)
        "boolean" -> BoolSpec
        "categorical" -> NominalSpec(labels)
        else -> error("validated variable type")
    }
}

@Serializable
internal data class ModelObjective(val minimize: String? = null, val maximize: String? = null) {
    val name: String get() = minimize ?: maximize!!
    val ascending: Boolean get() = minimize != null
    init { require((minimize == null) != (maximize == null)) { "specify minimize or maximize" } }
}

@Serializable
internal data class ModelConstraint(
    val terms: Map<String, Int>,
    val atMost: Int? = null,
    val atLeast: Int? = null,
    val equal: Int? = null,
) {
    fun expression(variables: Map<String, ModelVariable>): BoolExpr {
        require(terms.isNotEmpty() && listOfNotNull(atMost, atLeast, equal).size == 1) {
            "a linear constraint requires terms and exactly one of atMost, atLeast, equal"
        }
        val children = terms.map { (name, coefficient) ->
            val variable = requireNotNull(variables[name]) { "unknown constraint variable '$name'" }
            require(variable.role != "observed") { "constraints cannot depend on an unobserved outcome" }
            val ref = when (variable.type) {
                "integer" -> IntRef(name)
                "boolean" -> IntIfThenElse(BoolRef(name), IntLit(1), IntLit(0))
                else -> throw IllegalArgumentException("linear constraints require integer or Boolean variables")
            }
            IntScale(coefficient, ref)
        }
        val op = if (atMost != null) IntCmpOp.LE else if (atLeast != null) IntCmpOp.GE else IntCmpOp.EQ
        return IntCompare(IntSum(children), op, IntLit(atMost ?: atLeast ?: equal!!))
    }
}

@Serializable
internal data class ModelQuery(
    val given: JsonObject = emptyObject,
    val values: JsonObject = emptyObject,
    val count: Int = 1,
    val timeoutMillis: Long = 30_000,
) {
    init {
        validateTimeout(timeoutMillis)
        require(count in 1..64) { "count must be between 1 and 64" }
    }
}

@Serializable
internal data class ModelObservation(
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
