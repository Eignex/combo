package combo.math

import combo.test.assertEquals
import kotlin.test.Test

class DistributionsTest {
    @Test
    fun fCdf() {
        assertEquals(0.577f, fCdfDf1(1.0f, 2.0f), 0.1f) // Very inaccurate for low DF
        assertEquals(0.9726f, fCdfDf1(5.026f, 92.0f), 0.01f)
        assertEquals(0.8424f, fCdfDf1(2.0f, 1000.0f), 0.1f)
    }

    @Test
    fun normCdf() {
        assertEquals(0.0228f, normCdf(-2f), 0.001f)
        assertEquals(0.1587f, normCdf(-1f), 0.001f)
        assertEquals(0.5000f, normCdf(0f), 0.001f)
        assertEquals(0.8413f, normCdf(1f), 0.001f)
        assertEquals(0.9772f, normCdf(2f), 0.001f)
    }

    @Test
    fun invNormCdf() {
        assertEquals(-1.281f, normInvCdf(0.1f), 0.001f)
        assertEquals(-0.6745f, normInvCdf(0.25f), 0.001f)
        assertEquals(0.0f, normInvCdf(0.5f), 0.001f)
        assertEquals(1.2816f, normInvCdf(0.9f), 0.001f)
        assertEquals(2.3263f, normInvCdf(0.99f), 0.001f)
    }

    @Test
    fun chi2Cdf() {
        assertEquals(0.0f, chi2CdfDf1(0.0f), 0.0f)
        assertEquals(0.2481f, chi2CdfDf1(0.1f), 0.01f)
        assertEquals(0.5205f, chi2CdfDf1(0.5f), 0.01f)
        assertEquals(0.6572f, chi2CdfDf1(0.9f), 0.01f)
        assertEquals(0.6826f, chi2CdfDf1(1.0f), 0.01f)
        assertEquals(0.8427f, chi2CdfDf1(2.0f), 0.01f)
        assertEquals(0.9756f, chi2CdfDf1(5.0f), 0.01f)
        assertEquals(0.9984f, chi2CdfDf1(10.0f), 0.01f)
    }
}
