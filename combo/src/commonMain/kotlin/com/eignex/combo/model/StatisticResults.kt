@file:OptIn(kotlinx.serialization.InternalSerializationApi::class)

package com.eignex.combo.model

import com.eignex.kumulant.core.Result
import com.eignex.kumulant.core.ResultList
import com.eignex.skema.SchemaJson
import kotlinx.serialization.KSerializer
import kotlinx.serialization.PolymorphicSerializer
import kotlinx.serialization.json.*
import kotlinx.serialization.modules.SerializersModule
import kotlinx.serialization.serializer

private val resultJson = Json(SchemaJson) {
    allowSpecialFloatingPointValues = true
    serializersModule = SerializersModule {
        include(SchemaJson.serializersModule)
        polymorphicDefaultSerializer(Result::class, ::resultSerializer)
    }
}

@Suppress("UNCHECKED_CAST")
private fun resultSerializer(result: Result): KSerializer<Result> = when (result) {
    is ResultList<*> -> ResultList.serializer(PolymorphicSerializer(Result::class))
    else -> result::class.serializer()
} as KSerializer<Result>

/** Encode a native Kumulant result, preserving its structure and representing nonfinite values as null. */
fun encodeStatisticResult(result: Result): JsonElement =
    resultJson.encodeToJsonElement(resultSerializer(result), result).finiteNumbers()

// Undefined statistical values remain distinguishable without emitting non-standard JSON numbers.
private fun JsonElement.finiteNumbers(): JsonElement = when (this) {
    is JsonObject -> JsonObject(mapValues { (_, value) -> value.finiteNumbers() })
    is JsonArray -> JsonArray(map { it.finiteNumbers() })
    is JsonPrimitive -> if (!isString && doubleOrNull?.isFinite() == false) JsonNull else this
}
