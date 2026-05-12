package combo.math

import kotlin.test.Test
import kotlin.test.assertEquals

class NumbersTest {
    @Test
    fun gcd() {
        assertEquals(10, gcd(10, 0))
        assertEquals(9, gcd(0, 9))
        assertEquals(1, gcd(1, 1))
        assertEquals(3, gcd(9, 6))
        assertEquals(2, gcd(6, 4))
        assertEquals(5, gcd(10, 5))
        assertEquals(160, gcd(640, 480))
    }

    @Test
    fun gcdPrimes() {
        assertEquals(1, gcd(13, 23))
        assertEquals(1, gcd(2, 79))
        assertEquals(1, gcd(47, 19))
    }

    @Test
    fun gcdAll() {
        assertEquals(25, gcdAll(-100, 50, 25))
        assertEquals(1, gcdAll(-10, 11, 0, 13))
        assertEquals(2, gcdAll(6, 8, 4, 16))
    }
}