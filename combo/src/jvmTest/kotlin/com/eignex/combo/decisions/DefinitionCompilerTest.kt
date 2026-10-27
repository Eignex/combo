package com.eignex.combo.decisions

import com.eignex.combo.shared.HybridDescentSession
import com.eignex.combo.shared.LinearFeatureProjection
import com.eignex.combo.shared.WitnessRequest
import com.eignex.klause.model.BoolRef
import com.eignex.klause.model.BoolSpec
import com.eignex.klause.model.NominalSpec
import com.eignex.klause.model.IntSpec
import com.eignex.klause.schema.BoolHandle
import com.eignex.skema.SchemaJson
import kotlinx.serialization.encodeToString
import kotlin.math.ln
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class DefinitionCompilerTest {
    private class Child : SubSpace() {
        val flag by boolVar()
        val size by intVar(1, 5)
    }

    private class Model : DecisionSpace() {
        val child by optionalDecisionSpace(::Child)
        val flag by boolVar()
        val segment by contextInt(1, 10, scale = Scale.Log)
        val premium by optionalContextBool()
        val effect by interact(segment, flag)
    }

    @Test
    fun `wire compilation restores contexts interactions and activation conditions`() {
        val original = Model().compileSpace()
        val text = SchemaJson.encodeToString(original.definition)
        val restored = SchemaJson.decodeFromString<DecisionSpaceDef>(text).compileSpace()
        assertEquals(original.definition, restored.definition)
        assertEquals(original.activeConditions, restored.activeConditions)
        assertEquals(original.contextBools.map { it.name }, restored.contextBools.map { it.name })
        assertEquals(original.contextInts.map { it.name }, restored.contextInts.map { it.name })
        assertEquals(original.interactions.map { it.toDef() }, restored.interactions.map { it.toDef() })
        assertTrue(restored.contextBools.single().isOptional)
        assertTrue(restored.gates.isEmpty())
        val context = context { set(restored.contextInts.single(), 3) }
        val batch = HybridDescentSession.klause(restored).witnesses(
            WitnessRequest(3, restored.assumptionsFor(context)),
            Random(7),
        )
        assertTrue(batch.witnesses.isNotEmpty())
        val projection = LinearFeatureProjection(restored)
        for (sample in batch.witnesses) {
            val features = projection.encode(BanditSample(sample), context)
            val contextSlot = projection.layout.intStart + restored.compiled.intVarIdByName.getValue("segment")
            assertEquals(ln(3.0), features[contextSlot], 1e-12)
            val flag = restored.compiled.decode(BoolHandle("flag"), sample)
            assertEquals(if (flag) ln(3.0) else 0.0, features[projection.layout.interactionsStart], 1e-12)
            assertTrue(restored.matches(sample, restored.assumptionsFor(context)))
        }
    }

    @Test
    fun `compiled definitions enforce original constraints`() {
        val restored = DecisionSpaceDef(
            variables = mapOf("flag" to BoolSpec),
            constraints = mapOf("yes" to BoolRef("flag")),
        ).compileSpace()
        val witnesses = HybridDescentSession.klause(restored).witnesses(WitnessRequest(1), Random(3)).witnesses
        assertEquals(1, witnesses.size)
        assertTrue(restored.compiled.decode(BoolHandle("flag"), witnesses.single()))
    }

    @Test
    fun `unsupported context types and invalid interaction references are rejected`() {
        assertFailsWith<IllegalArgumentException> {
            DecisionSpaceDef(context = mapOf("category" to NominalSpec(listOf("a", "b")))).compileSpace()
        }
        assertFailsWith<IllegalArgumentException> {
            DecisionSpaceDef(
                variables = mapOf("size" to IntSpec(0, 1)),
                interactions = listOf(
                    InteractionDef(
                    "missing",
                    ScalarRef(ScalarKind.IntContext, listOf("context", "missing")),
                    ScalarRef(ScalarKind.IntDecision, listOf("size")),
                )
                ),
            ).compileSpace()
        }
    }
}
