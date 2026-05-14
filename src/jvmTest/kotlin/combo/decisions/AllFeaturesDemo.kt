package combo.decisions

import com.eignex.klause.ast.implies
import com.eignex.klause.ast.not
import com.eignex.skema.SchemaJson
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

private class AllFeatAudioBlock : SubSpace() {
    val mute by boolVar()
    val volume by intVar(0, 11)
    val codec by nominal("aac", "mp3", "opus")
}

private class AllFeatAdSlot : SubSpace() {
    val premium by boolVar()
    val budget by intVar(0, 1000)
    val type by nominal("a", "b", "c")
    val audio by optionalDecisionSpace(::AllFeatAudioBlock)
    val noPremiumForA by constraint { (type eq "a") implies !premium }
}

private class FullModel : DecisionSpace() {
    val baseline by boolVar()
    val tier by nominal("free", "pro", "enterprise")
    val slotA by decisionSpace(::AllFeatAdSlot)
    val slotB by decisionSpace(::AllFeatAdSlot)

    val premiumCtx by contextBool()
    val segment by contextInt(-100, 100)

    val premiumXslotABudget by interact(premiumCtx, slotA.budget)
    val premiumXtier by interact(premiumCtx, tier)
    val premiumXsegment by interact(premiumCtx, segment)
}

class AllFeaturesDemo {

    @Test
    fun `print full DecisionSpaceDef JSON`() {
        val model = FullModel()
        val def = model.definition()
        val json = SchemaJson.encodeToString(DecisionSpaceDef.serializer(), def)
        println(json)
    }

    @Test
    fun `DecisionSpaceDef should round-trip and recompile to the same constraint problem`() {
        val original = FullModel().definition()
        val encoded = SchemaJson.encodeToString(DecisionSpaceDef.serializer(), original)
        val decoded = SchemaJson.decodeFromString(DecisionSpaceDef.serializer(), encoded)

        assertEquals(original.name, decoded.name)
        assertEquals(original.variables.keys, decoded.variables.keys)
        assertEquals(original.spaces.keys, decoded.spaces.keys)
        assertEquals(original.context.keys, decoded.context.keys)
        assertEquals(
            original.interactions.map { it.name },
            decoded.interactions.map { it.name },
        )
        // slotA's audio sub-space should survive as an optional nested space.
        val slotA = decoded.spaces.getValue("slotA")
        assertTrue("audio" in slotA.optionalSpaces.keys)

        val recompiled = decoded.compile()
        val direct = original.compile()
        assertEquals(direct.problem.numBoolVars, recompiled.problem.numBoolVars)
        assertEquals(direct.problem.numIntVars, recompiled.problem.numIntVars)
    }
}
