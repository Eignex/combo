package com.eignex.combo.model

import com.eignex.combo.decisions.DecisionSpaceDef
import com.eignex.klause.model.AllDifferent
import com.eignex.klause.model.IntCmpOp
import com.eignex.klause.model.IntCompare
import com.eignex.klause.model.IntLit
import com.eignex.klause.model.IntMul
import com.eignex.klause.model.IntRef
import com.eignex.klause.model.IntSpec
import com.eignex.klause.model.NamedConstraint
import com.eignex.skema.SchemaJson
import kotlinx.serialization.encodeToString
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse

class ModelDefinitionTest {
    @Test
    fun `model contract retains native nested variables and constraints`() {
        val product = IntCompare(IntMul(IntRef("x"), IntRef("y")), IntCmpOp.EQ, IntLit(4))
        val distinct = AllDifferent(listOf(IntRef("x"), IntRef("y")))
        val child = DecisionSpaceDef(
            variables = mapOf("x" to IntSpec(1, 4), "y" to IntSpec(1, 4)),
            constraints = mapOf("product" to product, "distinct" to distinct),
        )
        val model = ModelDefinition(
            space = DecisionSpaceDef(spaces = mapOf("child" to child)),
            objective = ModelObjective(minimize = "child.x"),
        )
        val encoded = SchemaJson.encodeToString(model)
        assertFalse("\"type\"" in encoded)
        val restored = SchemaJson.decodeFromString<ModelDefinition>(encoded)
        assertEquals(model, restored)
        val entries = restored.space!!.toSchemaDef().entries
        assertEquals(
            NamedConstraint(IntCompare(IntMul(IntRef("child.x"), IntRef("child.y")), IntCmpOp.EQ, IntLit(4))),
            entries["child.product"],
        )
        assertEquals(
            NamedConstraint(AllDifferent(listOf(IntRef("child.x"), IntRef("child.y")))),
            entries["child.distinct"],
        )
    }

    @Test
    fun `measured outcome is separate from the search domain`() {
        val model = SchemaJson.decodeFromString<ModelDefinition>(
            """
            {"space":{"variables":{"workers":{"int":[1,8]}}},
             "outcome":"latency","objective":{"minimize":"latency"}}
            """.trimIndent(),
        )
        assertEquals(true, model.learned)
        assertEquals(setOf("workers"), model.space!!.toSchemaDef().entries.keys)
        assertEquals(model, SchemaJson.decodeFromString<ModelDefinition>(SchemaJson.encodeToString(model)))
        for (text in listOf(
            """{"space":{"variables":{"x":{"int":[1,8]}}},"objective":{"minimize":"typo"}}""",
            """{"space":{"variables":{"x":{"int":[1,8]}}},"outcome":"x","objective":{"minimize":"x"}}""",
            """{"space":{"variables":{"x":{"int":[1,8]}}},"outcome":"y"}""",
        )) {
            assertFailsWith<IllegalArgumentException> { SchemaJson.decodeFromString<ModelDefinition>(text) }
        }
    }
}
