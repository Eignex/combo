package combo.math

import kotlin.random.Random

fun permutation(size: Int = Int.MAX_VALUE, rng: Random = Random(0)): IntPermutation {
    if (size <= 1) return object : IntPermutation {
        override fun iterator(): IntIterator {
            return (0 until size).iterator()
        }

        override fun encode(value: Int) = 0
    }
    return CyclingHashIntPermutation(size, rng)
}

interface IntPermutation : Iterable<Int> {
    override fun iterator(): IntIterator
    fun encode(value: Int): Int
}

/**
 * Using format-preserving encryption with cycling.
 */
class CyclingHashIntPermutation(val size: Int = Int.MAX_VALUE, rng: Random) : IntPermutation {

    private val mask: Int // bit mask for block
    // 0 < size <= mask+1 and mask+1 is a power of 2
    private val rish: Int                 // right shift count
    private val rk1 = rng.nextInt()
    private val rk2 = rng.nextInt()
    private val rk3 = rng.nextInt()

    init {
        var i = 8
        var j = 3
        while (j < 31 && i < size) {
            i += i
            j++
        }
        this.mask = i - 1
        this.rish = j * 3 / 7
    }

    override fun encode(value: Int): Int {
        assert(value in 0 until size)
        var x = value
        do {
            val const = 0x7F4A7C15
            x = (x * const + rk1) and mask
            x = x xor (x ushr rish)
            x = (x * const + rk2) and mask
            x = x xor (x ushr rish)
            x = (x * const + rk3) and mask
            x = x xor (x ushr rish)
        } while (x >= this.size)
        return x
    }

    override fun iterator(): IntIterator {
        var i = 0
        return object : IntIterator() {
            override fun hasNext() = i < size
            override fun nextInt() = encode(i++)
        }
    }
}
