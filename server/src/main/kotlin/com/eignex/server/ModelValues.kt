package com.eignex.server

import com.eignex.combo.decisions.DecisionSpaceDef
import com.eignex.klause.model.NamedConstraint
import com.eignex.klause.model.mapReferences
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.Context
import com.eignex.combo.decisions.context
import com.eignex.combo.decisions.floatAssumptions
import com.eignex.klause.compile.CompiledSchema
import com.eignex.klause.model.BoolSpec
import com.eignex.klause.model.FloatSpec
import com.eignex.klause.model.IntSpec
import com.eignex.klause.model.NominalSpec
import com.eignex.klause.model.PresenceSpec
import com.eignex.klause.model.SchemaEntry
import com.eignex.klause.model.VarSpec
import com.eignex.klause.propagation.Assumptions
import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.FloatHandle
import com.eignex.klause.schema.IntHandle
import com.eignex.klause.schema.NominalHandle
import com.eignex.klause.solver.Sample
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.put

/** Check runtime support without introducing a separate variable or expression vocabulary. */
internal fun requireSupportedSpace(space: DecisionSpaceDef) {
    fun visit(node: DecisionSpaceDef, root: Boolean) {
        require(node.optionalContext.isEmpty()) { "optional context is not supported by this API yet" }
        require(root || node.context.isEmpty()) { "context must be declared at the root" }
        require(node.context.values.all { it == BoolSpec || it is IntSpec || it is FloatSpec }) {
            "context must be Boolean or numeric"
        }
        node.spaces.values.forEach { visit(it, false) }
        node.optionalSpaces.values.forEach { visit(it, false) }
    }
    visit(space, true)
    val entries = space.toSchemaDef().entries
    requireScalarVariables(entries)

}

internal fun requireScalarVariables(entries: Map<String, SchemaEntry>) {
    for ((name, entry) in entries) {
        require(entry !is VarSpec || entry == BoolSpec || entry is IntSpec || entry is FloatSpec ||
            entry is NominalSpec || entry is PresenceSpec) { "unsupported variable type at '$name'" }
    }
}

internal inline fun <T> compileInput(compile: () -> T): T = try {
    compile()
} catch (error: IllegalStateException) {
    throw IllegalArgumentException(error.message ?: "invalid model", error)
}

internal fun values(compiled: CompiledSchema, entries: Map<String, SchemaEntry>, sample: Sample): JsonObject =
    buildJsonObject {
        val assignment = sample.compatibilitySnapshot()
        for ((name, spec) in entries) {
            when (spec) {
                BoolSpec, is PresenceSpec -> put(name, compiled.decode(BoolHandle(name), assignment))
                is IntSpec -> put(name, compiled.decode(IntHandle(name, spec.min, spec.max), assignment))
                is NominalSpec -> put(name, compiled.decode(NominalHandle(name, spec.labels), assignment))
                is FloatSpec -> put(name, compiled.decode(FloatHandle(name, spec.min, spec.max), assignment))
                else -> Unit
            }
        }
    }

internal fun assumptions(compiled: CompiledSchema, entries: Map<String, SchemaEntry>, input: JsonObject): Assumptions {
    var result = Assumptions.None
    for ((name, element) in input) {
        val value = element.jsonPrimitive
        result = when (val spec = entries[name]) {
            BoolSpec, is PresenceSpec -> {
                require(!value.isString) { "'$name' must be Boolean" }
                result.withBool(compiled.boolVarIdByName.getValue(name), requireNotNull(value.booleanOrNull) { "'$name' must be Boolean" })
            }
            is IntSpec -> {
                require(!value.isString) { "'$name' must be an integer" }
                val number = value.long
                require(number in spec.min.toLong()..spec.max.toLong()) { "'$name' outside its bounds" }
                result.withInt(compiled.intVarIdByName.getValue(name), number)
            }
            is NominalSpec -> {
                require(value.isString && value.content in spec.labels) { "invalid label for '$name'" }
                var pinned = result
                for ((label, id) in compiled.nominalIndicators.getValue(name)) {
                    pinned = pinned.withBool(id, label == value.content)
                }
                pinned
            }
            is FloatSpec -> {
                val number = requireNotNull(value.takeUnless { it.isString }?.doubleOrNull?.takeIf { it.isFinite() }) {
                    "'$name' must be a finite number"
                }
                result.mergedWith(compiled.floatAssumptions(mapOf(FloatHandle(name, spec.min, spec.max) to number)))
            }
            else -> throw IllegalArgumentException("unknown variable '$name'")
        }
    }
    return result
}

internal fun readContext(space: CompiledDecisionSpace, input: JsonObject): Context {
    val names = (space.contextBools.map { it.name } + space.contextInts.map { it.name } + space.contextFloats.map { it.name }).toSet()
    require(input.keys.all { it in names }) { "unknown context field" }
    return context {
        for (handle in space.contextBools) {
            val value = input[handle.name]
            require(value != null || handle.isOptional) { "missing context '${handle.name}'" }
            if (value != null) {
                require(!value.jsonPrimitive.isString) { "context '${handle.name}' must be Boolean" }
                set(handle, requireNotNull(value.jsonPrimitive.booleanOrNull) { "context '${handle.name}' must be Boolean" })
            }
        }
        for (handle in space.contextFloats) {
            val value = requireNotNull(input[handle.name]) { "missing context '${handle.name}'" }.jsonPrimitive
            val number = requireNotNull(value.takeUnless { it.isString }?.doubleOrNull?.takeIf { it.isFinite() }) {
                "context '${handle.name}' must be a finite number"
            }
            set(handle, number)
        }
        for (handle in space.contextInts) {
            val value = input[handle.name]
            require(value != null || handle.isOptional) { "missing context '${handle.name}'" }
            if (value != null) {
                require(!value.jsonPrimitive.isString) { "context '${handle.name}' must be an integer" }
                val number = value.jsonPrimitive.int
                require(number in handle.min..handle.max) { "context '${handle.name}' outside its bounds" }
                set(handle, number)
            }
        }
    }
}
