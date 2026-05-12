package combo.decisions

import com.eignex.klause.ast.SchemaEntry
import com.eignex.klause.ast.implies
import com.eignex.klause.ast.not
import com.eignex.klause.compile.compile
import com.eignex.skema.SchemaDef
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
    val audio by optionalSubspace(::AllFeatAudioBlock)            // nested optional
    val noPremiumForA by constraint { (type eq "a") implies !premium }
}

private class FullModel : DecisionSpace() {
    val baseline by boolVar()
    val tier by nominal("free", "pro", "enterprise")
    val slotA by subspace(::AllFeatAdSlot)
    val slotB by subspace(::AllFeatAdSlot)

    val premiumCtx by contextBool()
    val segment by contextInt()

    val premiumXslotABudget by interact(premiumCtx, slotA.budget)
    val premiumXtier by interact(premiumCtx, tier)
    val premiumXsegment by interact(premiumCtx, segment)
}

class AllFeaturesDemo {

    @Test
    fun `print full DecisionSpaceDef JSON`() {
        val model = FullModel()
        val space = model.compileSpace()
        val def = space.definition()
        val json = SchemaJson.encodeToString(DecisionSpaceDef.serializer(), def)
        println(json)
    }

    @Test
    fun `DecisionSpaceDef round-trips and recovers a compilable klause schema`() {
        val model = FullModel()
        val original = model.compileSpace()
        val def = original.definition()

        // JSON → DecisionSpaceDef
        val encoded = SchemaJson.encodeToString(DecisionSpaceDef.serializer(), def)
        val decoded = SchemaJson.decodeFromString(DecisionSpaceDef.serializer(), encoded)

        // Klause schema matches.
        assertEquals(original.schemaDef.entries.keys, decoded.klause.entries.keys)

        // Context handles, interactions, gates round-trip.
        assertEquals(listOf("premiumCtx"), decoded.contextBools)
        assertEquals(listOf("segment"), decoded.contextInts)
        assertEquals(
            listOf("premiumXslotABudget", "premiumXtier", "premiumXsegment"),
            decoded.interactions.map { it.name },
        )
        assertTrue("slotA.audio" in decoded.gates)
        assertTrue("slotB.audio" in decoded.gates)

        // Klause is still compilable from the decoded schema.
        val klauseDef: SchemaDef<SchemaEntry> = decoded.klause
        // Build a throwaway VariableSchema from the decoded SchemaDef and compile it.
        val recompiled = recompile(klauseDef)
        assertEquals(original.compiled.problem.numBoolVars, recompiled.problem.numBoolVars)
        assertEquals(original.compiled.problem.numIntVars, recompiled.problem.numIntVars)
    }

    private fun recompile(def: SchemaDef<SchemaEntry>): com.eignex.klause.compile.CompiledProblem {
        // Reuse RootKlauseSchema (which extends VariableSchema) as the loader.
        val schema = RootKlauseSchema()
        for ((name, entry) in def.entries) {
            when (entry) {
                is com.eignex.klause.ast.BoolSpec -> schema.registerBool(name, null)
                is com.eignex.klause.ast.IntSpec -> schema.registerInt(name, entry.min, entry.max, null)
                is com.eignex.klause.ast.NominalSpec -> schema.registerNominal(name, entry.labels, null)
                is com.eignex.klause.ast.FloatSpec -> error("float roundtrip not exercised yet")
                is com.eignex.klause.ast.NamedConstraint -> schema.registerConstraint(name, entry.expr)
            }
        }
        return schema.compile()
    }
}
