package combo.util

import combo.test.assertContentEquals
import kotlin.test.Test
import kotlin.test.assertSame
import kotlin.test.assertTrue

class ArraysTest {
    @Test
    fun arrayTransformToEmptyIntArray() {
        assertTrue { EMPTY_INT_ARRAY.apply { transformArray { it + 1 } }.isEmpty() }
    }

    @Test
    fun arrayTransformToIntArray() {
        assertTrue { intArrayOf(1, 2, 3).contentEquals(intArrayOf(0, 1, 2).apply { transformArray { it + 1 } }) }
    }

    @Test
    fun arrayTransformToEmptyFloatArray() {
        assertTrue { floatArrayOf().apply { transformArray { Float.POSITIVE_INFINITY } }.isEmpty() }
    }

    @Test
    fun arrayTransformToFloatArray() {
        assertTrue { floatArrayOf(1.0f, 2.0f, 3.0f).contentEquals(floatArrayOf(2.0f, 4.0f, 6.0f).apply { transformArray { it / 2.0f } }) }
    }

    @Test
    fun shouldNotCopyArray() {
        val array = IntArray(10)
        assertSame(array, array.apply { transformArray { it + 1 } })
    }

    @Test
    fun removeOne() {
        assertTrue(intArrayOf(1).removeAt(0).isEmpty())
    }

    @Test
    fun removeLast() {
        assertContentEquals(intArrayOf(1, 2, 3), intArrayOf(1, 2, 3, 4).removeAt(3))
    }

    @Test
    fun removeMiddle() {
        assertContentEquals(intArrayOf(1, 3, 4), intArrayOf(1, 2, 3, 4).removeAt(1))
        assertContentEquals(intArrayOf(1, 2, 4), intArrayOf(1, 2, 3, 4).removeAt(2))
    }

    @Test
    fun removeFirst() {
        assertContentEquals(intArrayOf(2, 3, 4), intArrayOf(1, 2, 3, 4).removeAt(0))
    }
}
