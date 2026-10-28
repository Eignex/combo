package com.eignex.combo.decisions

import com.eignex.klause.compile.CompiledSchema
import com.eignex.klause.ir.Problem
import com.eignex.klause.model.FloatSpec
import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.FloatHandle
import com.eignex.klause.schema.IntHandle
import com.eignex.klause.schema.NominalHandle
import com.eignex.klause.solver.Sample
import com.eignex.klause.util.toLongExact
import com.eignex.klause.solver.objective.LinearObjective

/** Optional schema names over unchanged native problem columns; contains no variable or factor definitions. */
class DecisionCoordinates private constructor(
    /** The original native model, including its domains, factors, settings, and semantics. */
    val problem: Problem,
    /** Original name/decoder metadata when this model was compiled from a schema. */
    val schema: CompiledSchema?,
) {
    /** Boolean column ids, using canonical `bools.index` names for an imported problem. */
    val boolVarIdByName: Map<String, Int> = schema?.boolVarIdByName
        ?: (0 until problem.numBoolVars).associateBy { "bools.$it" }

    /** Integer column ids; their native domains remain Long-valued, sparse, or open. */
    val intVarIdByName: Map<String, Int> = schema?.intVarIdByName
        ?: (0 until problem.numIntVars).associateBy { "ints.$it" }

    /** Existing nominal indicators; raw problems expose the underlying Boolean columns. */
    val nominalIndicators: Map<String, Map<String, Int>> = schema?.nominalIndicators.orEmpty()

    /** Existing schema float decoders; raw IR reals never acquire a synthetic grid. */
    val floatDecoders: Map<String, FloatSpec> = schema?.floatDecoders.orEmpty()

    /** Real columns in native units, with their original open or finite bounds. */
    val realHandles: Map<Int, FloatHandle> = if (schema == null) {
        (0 until problem.numRealVars).associateWith {
            FloatHandle("reals.$it", problem.realLower[it], problem.realUpper[it])
        }
    } else {
        schema.floatDecoders.mapNotNull { (name, spec) ->
            val handle = FloatHandle(name, spec.min, spec.max)
            val id = schema.minimize(handle).realCoefficients.indexOfFirst { it != 0.0 }
            if (id < 0) null else id to handle
        }.toMap()
    }

    /** Decode through the original schema, or read the unchanged native Boolean column. */
    fun decode(handle: BoolHandle, sample: Sample): Boolean =
        schema?.decode(handle, sample) ?: sample.boolValue(boolVarIdByName.getValue(handle.name))

    /** Decode through the original schema, or read the unchanged native integer column. */
    fun decode(handle: IntHandle, sample: Sample): Long =
        schema?.decode(handle, sample) ?: sample.exactIntValue(intVarIdByName.getValue(handle.name)).toLongExact()

    /** Decode through the original schema, or read the native real approximation for model features. */
    fun decode(handle: FloatHandle, sample: Sample): Double = schema?.decode(handle, sample)
        ?: sample.approximateRealValue(realHandles.entries.single { it.value.name == handle.name }.key)

    /** Decode existing categorical metadata without reconstructing a schema for raw problems. */
    fun decode(handle: NominalHandle, sample: Sample): String =
        requireNotNull(schema) { "native problems expose nominal indicators as Boolean columns" }.decode(handle, sample)

    /** Native objective for an existing Boolean handle. */
    fun minimize(handle: BoolHandle): LinearObjective = schema?.minimize(handle) ?: objective(handle.name)

    /** Native objective for an existing integer handle. */
    fun minimize(handle: IntHandle): LinearObjective = schema?.minimize(handle) ?: objective(handle.name)

    /** Native objective for an existing real handle. */
    fun minimize(handle: FloatHandle): LinearObjective = schema?.minimize(handle) ?: objective(handle.name)

    /** Minimize a native column without imposing Int-sized bounds or finite real ranges. */
    fun objective(name: String): LinearObjective = when {
        name in boolVarIdByName -> LinearObjective(
            boolWeights = LongArray(problem.numBoolVars).also { it[boolVarIdByName.getValue(name)] = 1 },
        )
        name in intVarIdByName -> LinearObjective(
            intCoefficients = LongArray(problem.numIntVars).also { it[intVarIdByName.getValue(name)] = 1 },
        )
        else -> {
            val id = realHandles.entries.firstOrNull { it.value.name == name }?.key
            requireNotNull(id) { "unknown numeric column '$name'" }
            LinearObjective(realCoefficients = DoubleArray(problem.numRealVars).also { it[id] = 1.0 })
        }
    }

    /** Coordinate metadata constructors; neither changes the native model. */
    companion object {
        /** Keep an existing compiled schema's names and decoders. */
        fun of(schema: CompiledSchema): DecisionCoordinates = DecisionCoordinates(schema.problem, schema)

        /** Index a complete native problem directly, without a schema compilation or factor translation. */
        fun of(problem: Problem): DecisionCoordinates = DecisionCoordinates(problem, null)
    }
}
