package combo.sat

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class DimacsFormatKtTest {

    @Test
    fun toLiteral() {
        assertEquals(1, 0.toLiteral(true))
        assertEquals(-1, 0.toLiteral(false))

        assertEquals(11, 10.toLiteral(true))
        assertEquals(-11, 10.toLiteral(false))
    }

    @Test
    fun toId() {
        assertEquals(0, 1.toIx())
        assertEquals(0, (-1).toIx())
        assertEquals(1, (-2).toIx())
        assertEquals(3, 4.toIx())
        assertEquals(3, (-4).toIx())
    }

    @Test
    fun toBoolean() {
        assertTrue(1.toBoolean())
        assertFalse((-1).toBoolean())
        assertTrue(6.toBoolean())
        assertFalse((-7).toBoolean())
    }

    @Test
    fun not() {
        assertEquals(-1, !1)
        assertEquals(1, !(-1))
        assertEquals(-10, !10)
        assertEquals(-11, !11)
        assertEquals(11, !(!11))
    }
}
