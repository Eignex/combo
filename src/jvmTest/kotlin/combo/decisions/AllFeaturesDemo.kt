package combo.decisions

import com.eignex.klause.ast.SchemaEntry
import com.eignex.klause.ast.implies
import com.eignex.klause.ast.not
import com.eignex.skema.SchemaDef
import com.eignex.skema.SchemaJson
import kotlin.test.Test

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
    val slotA by subspace(::AllFeatAdSlot)                         // typed sub-space
    val slotB by subspace(::AllFeatAdSlot)                         // multi-instance

    val premiumCtx by contextBool()                         // context handles
    val segment by contextInt()

    val premiumXslotABudget by interact(premiumCtx, slotA.budget)   // ctx × decision
    val premiumXtier by interact(premiumCtx, tier)                  // ctx × nominal (3 slots)
    val premiumXsegment by interact(premiumCtx, segment)            // ctx × ctx
}

class AllFeaturesDemo {
    @Test
    fun `serialize full schema to JSON`() {
        val model = FullModel()
        val space = model.compileSpace()
        val def: SchemaDef<SchemaEntry> = space.schemaDef
        val json = SchemaJson.encodeToString(SchemaDef.serializer(SchemaEntry.serializer()), def)
        println(json)
    }
}
