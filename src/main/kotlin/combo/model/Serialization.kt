package combo.model

import combo.expressions.Constraint
import combo.expressions.Variable
import kotlinx.serialization.Serializable

data class Serialization(
    val variables: List<Variable<*, *>>,
    val constraints: List<Constraint>
)

@Serializable
data class ConstraintDto(
    val type: String,
)

@Serializable
data class VariableDto(
    val type: String,
    val cost: Int
)
