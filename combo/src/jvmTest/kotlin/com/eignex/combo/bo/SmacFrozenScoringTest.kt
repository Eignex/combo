package com.eignex.combo.bo

import com.eignex.combo.test.TimedTest
import java.nio.file.Files
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class SmacFrozenScoringTest : TimedTest() {
    @Test
    fun `raw reference predictions use Kumulant losses across groups and top regions`() {
        val directory = Files.createTempDirectory("combo-smac-scores").toFile()
        try {
            val input = directory.resolve("predictions.csv")
            val output = directory.resolve("models.csv")
            input.writeText(
                "problem,seed,history,prediction,variance,truth\n" +
                    "quadratic,201,16,1,0,1\n" +
                    "quadratic,201,16,4,1,2\n" +
                    "quadratic,201,16,6,4,3\n" +
                    "noisy,202,64,4,1,0\n" +
                    "noisy,202,64,2,0,2\n",
            )

            scoreSmacPredictions(input, output)

            val rows = output.readLines().drop(1).map { it.split(',') }
            assertEquals(2, rows.size)
            assertEquals(listOf("quadratic", "201", "16", "smac-2.4.1"), rows[0].take(4))
            val expected = listOf(
                listOf(sqrt(13.0 / 3.0), 1.0, 1.0, 0.0, 1.0),
                listOf(sqrt(8.0), 0.5, 0.5, 4.0, 0.0),
            )
            for (i in rows.indices) {
                for (j in expected[i].indices) {
                    assertEquals(expected[i][j], rows[i][j + 4].toDouble(), 1e-12)
                }
            }
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun `invalid variance is rejected before reporting reference losses`() {
        val directory = Files.createTempDirectory("combo-smac-scores").toFile()
        try {
            val input = directory.resolve("predictions.csv")
            input.writeText("problem,seed,history,prediction,variance,truth\nquadratic,201,16,1,-1,1\n")

            assertFailsWith<IllegalArgumentException> { scoreSmacPredictions(input, directory.resolve("models.csv")) }
        } finally {
            directory.deleteRecursively()
        }
    }
}
