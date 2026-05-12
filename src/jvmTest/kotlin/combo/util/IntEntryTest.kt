package combo.util

import kotlin.test.Test
import kotlin.test.assertEquals

class IntEntryTest {

    private fun test(key: Int, value: Int) {
        val entry = entry(key, value)
        assertEquals(key, entry.key(), "$key : $value")
        assertEquals(value, entry.value(), "$key : $value")
    }

    @Test
    fun keyValue() {
        for (i in -10..10) {
            test(i, i)
            test(i, -i)
        }
    }

    @Test
    fun keyValueBounds() {
        val values = intArrayOf(0, -1, Int.MAX_VALUE, Int.MIN_VALUE)
        for (v1 in values)
            for (v2 in values)
                test(v1, v2)
    }
}

