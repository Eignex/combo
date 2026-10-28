package com.eignex.combo.model

import kotlinx.serialization.json.*

/** Read an existing native result field; this performs no estimation or distribution sampling. */
internal fun StatisticalInput.readProjection(snapshots: Map<String, JsonElement>, projection: String): Double {
    return readPrimitive(snapshots, projection).double
}

internal fun StatisticalInput.readPrimitive(snapshots: Map<String, JsonElement>, projection: String): JsonPrimitive {
    val root = requireNotNull(snapshots[statistic]) { "missing snapshot '$statistic'" }
    val source = root.at(path)
    val primitive = source.at(projections[projection] ?: listOf(projection)) as? JsonPrimitive
    requireNotNull(primitive?.takeUnless { it.isString }?.doubleOrNull?.takeIf { it.isFinite() }) {
        "'$statistic.$projection' must select a finite numeric result; " +
            "available fields: ${(source as? JsonObject)?.keys.orEmpty()}"
    }
    return requireNotNull(primitive)
}

private fun JsonElement.at(path: List<String>): JsonElement = path.fold(this) { node, part ->
    when (node) {
        is JsonObject -> node[part]
        is JsonArray -> part.toIntOrNull()?.let { node.getOrNull(it) }
        else -> null
    } ?: throw IllegalArgumentException("missing statistic result path '${path.joinToString("/")}'")
}
