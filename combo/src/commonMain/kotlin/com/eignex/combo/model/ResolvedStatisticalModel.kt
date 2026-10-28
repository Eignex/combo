package com.eignex.combo.model

import com.eignex.klause.model.*
import com.eignex.klause.simplex.exact.BigFraction
import com.eignex.klause.util.BIG_ONE
import com.eignex.klause.util.BIG_ZERO
import com.eignex.klause.util.BigInt
import com.eignex.klause.util.bigIntOf
import com.eignex.klause.util.negate
import com.eignex.klause.util.plus
import com.eignex.klause.util.signum
import com.eignex.klause.util.times
import com.eignex.skema.SchemaDef
import kotlinx.serialization.json.*

/** Native Klause model with one frozen set of statistical input values. */
class ResolvedStatisticalModel internal constructor(
    /** Ordinary Klause declarations and constraints, ready for its unchanged compiler. */
    val schema: SchemaDef<SchemaEntry>,
    /** Objective rewritten to a fixed input declaration when necessary. */
    val objective: ModelObjective?,
    /** Selected projections, useful for explaining the statistical basis of a decision. */
    val inputs: JsonObject,
)

/**
 * Lower bindings using snapshots encoded with the native Kumulant result serializers.
 * Linear interval comparisons use their worst endpoint; nonlinear interval uses require
 * explicit projections. The result contains only existing Klause node types.
 */
fun ModelDefinition.resolveStatistics(snapshots: Map<String, JsonElement>): ResolvedStatisticalModel =
    StatisticalLowering(this, snapshots).resolve()

private class StatisticalLowering(
    private val model: ModelDefinition,
    private val snapshots: Map<String, JsonElement>,
) {
    private val space = requireNotNull(model.space) { "a decision space is required" }
    private val entries = LinkedHashMap(space.toSchemaDef().entries)
    private val constants = linkedMapOf<Pair<String, Boolean>, String>()
    private val selected = linkedMapOf<String, JsonElement>()
    private val intervals = hashMapOf<String, Pair<Double, Double>>()

    fun resolve(): ResolvedStatisticalModel {
        for ((name, entry) in entries.toMap()) {
            if (entry is NamedConstraint) entries[name] = entry.copy(expr = lower(entry.expr))
        }
        val goal = model.objective?.let { goal ->
            val name = scalar(goal.name, real = true)
            if (goal.ascending) ModelObjective(minimize = name) else ModelObjective(maximize = name)
        }
        return ResolvedStatisticalModel(SchemaDef(entries), goal, JsonObject(selected))
    }

    private fun binding(name: String): Map.Entry<String, StatisticalInput>? =
        model.inputs.entries.firstOrNull { name == it.key || name.startsWith("${it.key}.") }

    private fun uncertain(name: String): Boolean = model.inputs[name]?.value == null && name in model.inputs

    private fun uncertain(expr: BoolExpr): Boolean {
        var found = false
        expr.mapReferences { name ->
            found = found || uncertain(name)
            name
        }
        return found
    }

    private fun lower(expr: BoolExpr): BoolExpr = when (expr) {
        is And -> expr.copy(children = expr.children.map(::lower))

        is Or -> {
            require(!uncertain(expr)) { "uncertain disjunctions require explicit projections" }
            expr.copy(children = expr.children.map(::lower))
        }

        is IntCompare -> integerComparison(expr)

        is FloatLinearConstraint -> realComparison(expr)

        is ContinuousRealEquality -> {
            require(!uncertain(expr)) { PROJECTION_MESSAGE }
            expr.mapReferences { scalar(it, real = true) }
        }

        is Implies -> {
            require(!uncertain(expr.left)) { "uncertain implication conditions require explicit projections" }
            expr.copy(left = lower(expr.left), right = lower(expr.right))
        }

        is Not -> when (val child = expr.child) {
            is IntCompare -> integerComparison(child.copy(op = child.op.negated()))

            is FloatLinearConstraint -> realComparison(child.copy(op = child.op.negated()))

            else -> {
                require(!uncertain(child)) { "negated compound uncertainty requires explicit projections" }
                expr.copy(child = lower(child))
            }
        }

        else -> {
            require(!uncertain(expr)) { "this constraint requires explicit statistical projections" }
            expr.mapReferences { scalar(it, real = false) }
        }
    }

    private fun integerComparison(expr: IntCompare): BoolExpr {
        if (!uncertain(expr)) return expr.mapReferences { scalar(it, real = false) }
        val coefficients = linkedMapOf<String, BigInt>()
        affine(expr.left, BIG_ONE, coefficients)
        affine(expr.right, BIG_ONE.negate(), coefficients)
        return endpoints(expr.op) { maximize, operation ->
            expr.copy(op = operation).mapReferences { name ->
                val projection = endpoint(name, coefficients[name]?.signum() ?: 0, maximize)
                scalar(projection, real = false)
            }
        }
    }

    private fun affine(expr: IntExpr, factor: BigInt, out: MutableMap<String, BigInt>) {
        when (expr) {
            is IntLit -> Unit

            is IntRef -> if (uncertain(expr.name)) out[expr.name] = (out[expr.name] ?: BIG_ZERO) + factor

            is IntScale -> affine(expr.child, factor * bigIntOf(expr.coeff.toLong()), out)

            is IntSum -> expr.children.forEach { affine(it, factor, out) }

            is IntMul -> when {
                expr.left is IntLit -> affine(
                    expr.right,
                    factor * bigIntOf((expr.left as IntLit).value.toLong()),
                    out,
                )

                expr.right is IntLit -> affine(
                    expr.left,
                    factor * bigIntOf((expr.right as IntLit).value.toLong()),
                    out,
                )

                else -> throw IllegalArgumentException(PROJECTION_MESSAGE)
            }

            else -> throw IllegalArgumentException(PROJECTION_MESSAGE)
        }
    }

    private fun realComparison(expr: FloatLinearConstraint): BoolExpr {
        require(expr.coeffs.all { it.isFinite() } && expr.bound.isFinite()) {
            "linear coefficients and bounds must be finite"
        }
        if (!uncertain(expr)) return expr.mapReferences { scalar(it, real = true) }
        val coefficients = linkedMapOf<String, BigFraction>()
        for ((name, coefficient) in expr.varNames.zip(expr.coeffs.toList())) {
            coefficients[name] = (coefficients[name] ?: BigFraction.ZERO) + requireNotNull(
                BigFraction.ofDouble(coefficient),
            )
        }
        return endpoints(expr.op) { maximize, operation ->
            expr.copy(
                op = operation,
                varNames = expr.varNames.map { name ->
                    scalar(endpoint(name, coefficients.getValue(name).signum(), maximize), real = true)
                },
            )
        }
    }

    private fun endpoint(name: String, sign: Int, maximize: Boolean): String {
        if (!uncertain(name)) return name
        interval(name)
        return "$name.${if (sign != 0 && (sign > 0) == maximize) "upper" else "lower"}"
    }

    private fun scalar(name: String, real: Boolean): String {
        val (inputName, input) = binding(name) ?: return name
        val projection = if (name == inputName) {
            requireNotNull(input.value) { "select an explicit projection of '$name'" }
        } else {
            name.removePrefix("$inputName.")
        }
        val value = output(inputName, input, projection)
        return constants.getOrPut(name to real) {
            var generated = "__stat_${constants.size}"
            while (generated in entries) generated += "_"
            entries[generated] = if (real) {
                FloatSpec(value, value)
            } else {
                require(value.toInt().toDouble() == value) {
                    "'$name' is $value; integer expressions require an exact 32-bit integer projection"
                }
                IntSpec(value.toInt(), value.toInt())
            }
            generated
        }
    }

    private fun output(name: String, input: StatisticalInput, projection: String): Double {
        val value = input.readProjection(snapshots, projection)
        selected["$name.$projection"] = JsonPrimitive(value)
        return value
    }

    private fun interval(name: String): Pair<Double, Double> = intervals.getOrPut(name) {
        val input = model.inputs.getValue(name)
        val lower = output(name, input, "lower")
        val upper = output(name, input, "upper")
        require(lower <= upper) { "'$name' has an inverted interval" }
        lower to upper
    }

    private companion object {
        const val PROJECTION_MESSAGE =
            "nonlinear uncertainty requires an explicit projection such as .lower, .upper, .mean, or .median"
    }
}

private fun endpoints(op: IntCmpOp, build: (Boolean, IntCmpOp) -> BoolExpr): BoolExpr = when (op) {
    IntCmpOp.LT, IntCmpOp.LE -> build(true, op)
    IntCmpOp.GT, IntCmpOp.GE -> build(false, op)
    IntCmpOp.EQ -> And(listOf(build(false, op), build(true, op)))
    IntCmpOp.NE -> Or(listOf(build(true, IntCmpOp.LT), build(false, IntCmpOp.GT)))
}

private fun IntCmpOp.negated(): IntCmpOp = when (this) {
    IntCmpOp.LT -> IntCmpOp.GE
    IntCmpOp.LE -> IntCmpOp.GT
    IntCmpOp.GT -> IntCmpOp.LE
    IntCmpOp.GE -> IntCmpOp.LT
    IntCmpOp.EQ -> IntCmpOp.NE
    IntCmpOp.NE -> IntCmpOp.EQ
}
