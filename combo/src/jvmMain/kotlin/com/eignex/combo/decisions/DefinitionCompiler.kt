package com.eignex.combo.decisions

import com.eignex.klause.model.And
import com.eignex.klause.model.BoolExpr
import com.eignex.klause.model.BoolRef
import com.eignex.klause.model.BoolSpec
import com.eignex.klause.model.IntSpec
import com.eignex.klause.model.NominalSpec
import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.IntHandle
import com.eignex.klause.schema.NominalHandle

/**
 * Compile a wire definition, restoring context handles, interactions, and activation conditions.
 * Sub-space object gates are empty because a wire definition has no Kotlin [SubSpace] instances.
 */
fun DecisionSpaceDef.compileSpace(): CompiledDecisionSpace = DefinitionCompiler(this).compile()

private class DefinitionCompiler(private val definition: DecisionSpaceDef) {
    private val bools = linkedMapOf<String, BoolContextHandle>()
    private val ints = linkedMapOf<String, IntContextHandle>()
    private val floats = linkedMapOf<String, FloatContextHandle>()
    private val active = linkedMapOf<String, BoolExpr>()
    private val entries = definition.toSchemaDef().entries

    fun compile(): CompiledDecisionSpace {
        visit(definition, "", emptyList())
        val interactions = definition.interactions.map {
            val lhs = resolve(it.lhs)
            val rhs = resolve(it.rhs)
            require(isAllowedInteraction(lhs, rhs)) { "unsupported interaction '${it.name}'" }
            InteractionHandle(it.name, lhs, rhs)
        }
        return CompiledDecisionSpace(
            definition.compile(),
            bools.values.toList(),
            ints.values.toList(),
            interactions,
            definition,
            active,
            emptyMap(),
            floats.values.toList(),
        )
    }

    private fun visit(node: DecisionSpaceDef, prefix: String, gates: List<String>) {
        if (prefix.isNotEmpty()) {
            require(node.context.isEmpty() && node.optionalContext.isEmpty() && node.interactions.isEmpty()) {
                "context and interactions must be declared at the root"
            }
        }
        fun activate(name: String, extra: String? = null) {
            val conditions = gates + listOfNotNull(extra)
            if (conditions.isNotEmpty()) {
                active[name] = if (conditions.size == 1) {
                    BoolRef(conditions.single())
                } else {
                    And(conditions.map(::BoolRef))
                }
            }
        }
        for (name in node.variables.keys) activate(prefix + name)
        for ((name, labels) in node.multiples) for (label in labels) activate("$prefix$name.$label")
        for (name in node.optionalVariables.keys) {
            val qualified = prefix + name
            val gate = gateNameFor(qualified)
            activate(gate)
            activate(qualified, gate)
        }
        for ((name, spec) in node.context + node.optionalContext) {
            val gate = if (name in node.optionalContext) BoolHandle(gateNameFor(name)) else null
            when (spec) {
                BoolSpec -> bools[name] = BoolContextHandle(BoolHandle(name), gate)
                is IntSpec -> ints[name] = IntContextHandle(IntHandle(name, spec.min.toLong(), spec.max.toLong()), gate)
                is com.eignex.klause.model.FloatSpec -> {
                    require(gate == null) { "optional float context is not supported yet" }
                    floats[name] = FloatContextHandle(
                        com.eignex.klause.schema.FloatHandle(name, spec.min, spec.max), spec.buckets,
                    )
                }
                else -> throw IllegalArgumentException("context '$name' must be Boolean or numeric")
            }
            activate(name, gate?.name)
        }
        for ((name, child) in node.spaces) visit(child, "$prefix$name.", gates)
        for ((name, child) in node.optionalSpaces) {
            val gate = prefix + name
            activate(gate)
            visit(child, "$gate.", gates + gate)
        }
    }

    private fun resolve(ref: ScalarRef): Any {
        val name = ref.path.joinToString(".")
        val spec = entries[name]
        return when (ref.kind) {
            ScalarKind.BoolContext -> requireNotNull(bools[contextName(ref)]) { "unknown context $ref" }

            ScalarKind.IntContext -> requireNotNull(ints[contextName(ref)]) { "unknown context $ref" }

            ScalarKind.BoolDecision -> {
                require(spec == BoolSpec && name !in bools) { "unknown Boolean decision '$name'" }
                BoolHandle(name)
            }

            ScalarKind.IntDecision -> {
                require(spec is IntSpec && name !in ints) { "unknown integer decision '$name'" }
                IntHandle(name, spec.min.toLong(), spec.max.toLong())
            }

            ScalarKind.NominalDecision -> {
                require(spec is NominalSpec && spec.labels == ref.labels) { "unknown nominal decision '$name'" }
                NominalHandle(name, spec.labels)
            }

            ScalarKind.FloatContext -> requireNotNull(floats[contextName(ref)]) { "unknown context $ref" }

            ScalarKind.FloatDecision -> {
                require(spec is com.eignex.klause.model.FloatSpec && name !in floats) {
                    "unknown float decision '$name'"
                }
                com.eignex.klause.schema.FloatHandle(name, spec.min, spec.max)
            }

            ScalarKind.NominalContext -> throw IllegalArgumentException("nominal contexts are unsupported")
        }
    }

    private fun contextName(ref: ScalarRef): String {
        require(ref.path.size == 2 && ref.path.first() == "context") { "invalid context reference $ref" }
        return ref.path.last()
    }
}
