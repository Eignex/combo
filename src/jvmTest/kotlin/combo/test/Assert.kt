package combo.test

import kotlin.math.abs
import kotlin.test.assertTrue

fun assertEquals(expected: Float, actual: Float, epsilon: Float, message: String? = null) {
    if (actual == expected) return // prevents NaN/Inf. errors
    val abs = abs(expected - actual)
    assertTrue(abs <= epsilon, (message ?: "") + "\n" +
            "Expected :$expected\n" +
            "Actual   :$actual\n" +
            "Err      :$abs > $epsilon")
}

fun assertContentEquals(expected: FloatArray, actual: FloatArray, epsilon: Float = 0.0f) {
    for (i in actual.indices)
        assertEquals(expected[i], actual[i], epsilon)
}

fun assertContentEquals(expected: Array<FloatArray>, actual: Array<FloatArray>, epsilon: Float = 0.0f) {
    for (i in actual.indices)
        for (j in actual[i].indices)
            assertEquals(expected[i][j], actual[i][j], epsilon)
}

fun assertContentEquals(expected: Array<*>, actual: Array<*>, message: String? = null) {
    assertTrue(expected.contentEquals(actual), message
            ?: "Expected \n${actual.joinToString(",", "<", ">")} to equal \n${expected.joinToString(",", "<", ">")}")
}

fun assertContentEquals(expected: IntArray, actual: IntArray, message: String? = null) {
    assertTrue(expected.contentEquals(actual), message
            ?: "Expected \n${actual.joinToString(",", "<", ">")} to equal \n${expected.joinToString(",", "<", ">")}")
}

fun assertContentEquals(expected: LongArray, actual: LongArray, message: String? = null) {
    assertTrue(expected.contentEquals(actual), message
            ?: "Expected \n${actual.joinToString(",", "<", ">")} to equal \n${expected.joinToString(",", "<", ">")}")
}

inline fun <reified T> assertContentEquals(expected: List<T>, actual: List<T>, message: String? = null) {
    assertTrue(expected.toTypedArray().contentEquals(actual.toTypedArray()), message
            ?: "Expected \n${actual.joinToString(",", "<", ">")} to equal \n${expected.joinToString(",", "<", ">")}")
}
