package combo.sat

import combo.util.IntHashSet
import combo.util.isNotEmpty
import kotlin.math.pow
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals

class InstancePermutationTest {

    @Test
    fun emptySequence() {
        assertEquals(1, InstancePermutation(0, BitArrayFactory, Random).asSequence().count())
    }

    @Test
    fun sequenceSize() {
        assertEquals(2.0.pow(3).toInt(), InstancePermutation(3, BitArrayFactory, Random).asSequence().count())
        assertEquals(2.0.pow(4).toInt(), InstancePermutation(4, BitArrayFactory, Random).asSequence().count())
    }

    @Test
    fun noRepetition() {
        val list = InstancePermutation(4, BitArrayFactory, Random).asSequence().toList()
        val set = InstancePermutation(4, BitArrayFactory, Random).asSequence().toSet()
        assertEquals(set.size, list.size)
    }

    @Test
    fun takeMany() {
        assertEquals(16, InstancePermutation(4, BitArrayFactory, Random).asSequence().take(1000).toList().size)
    }

    @Test
    fun largePermuation() {
        val left = IntHashSet(nullValue = -1)
        val size = 1001
        left.addAll(0 until size)
        var k = 0
        val rng = Random(0)
        while (left.isNotEmpty() && k++ <= 10000) {
            val instance = InstancePermutation(size, BitArrayFactory, rng).asSequence().first()
            for (i in instance)
                left.remove(i)
        }
    }
}
