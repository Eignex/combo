package com.eignex.combo.model

import com.eignex.klause.formats.json.ProblemJson
import com.eignex.klause.ir.Problem
import kotlinx.serialization.KSerializer
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.json.JsonDecoder
import kotlinx.serialization.json.JsonEncoder
import kotlinx.serialization.json.JsonObject

/** Embed Klause's canonical problem document unchanged in a model definition. */
object NativeProblemSerializer : KSerializer<Problem> {
    override val descriptor: SerialDescriptor = JsonObject.serializer().descriptor

    override fun serialize(encoder: Encoder, value: Problem) {
        require(encoder is JsonEncoder) { "native problems require JSON" }
        encoder.encodeJsonElement(ProblemJson.default.encodeToJsonElement(value))
    }

    override fun deserialize(decoder: Decoder): Problem {
        require(decoder is JsonDecoder) { "native problems require JSON" }
        return ProblemJson.default.decodeFromJsonElement(decoder.decodeJsonElement())
    }
}
