package combo.util

import com.eignex.kpermute.intPermutation
import kotlin.random.Random

class IntUnionCollection(val a: IntCollection, val b: IntCollection) : IntCollection {
    override val size = a.size + b.size
    override fun copy() = IntUnionCollection(a.copy(), b.copy())
    override fun contains(value: Int) = a.contains(value) || b.contains(value)

    override fun map(transform: (Int) -> Int) = IntUnionCollection(a.map(transform), b.map(transform))

    override fun iterator(): IntIterator {
        val aItr = a.iterator()
        val bItr = b.iterator()
        return object : IntIterator() {
            override fun hasNext() = aItr.hasNext() || bItr.hasNext()
            override fun nextInt(): Int {
                return if (aItr.hasNext()) {
                    aItr.nextInt()
                } else {
                    bItr.nextInt()
                }
            }
        }
    }

    override fun permutation(rng: Random): IntIterator {
        return object : IntIterator() {
            val perm = intPermutation(size, rng)
            val aPerm = a.permutation(rng)
            val bPerm = b.permutation(rng)
            var i = 0
            override fun hasNext() = i < size
            override fun nextInt(): Int {
                return if (perm.encode(i++) < a.size) {
                    aPerm.nextInt()
                } else {
                    bPerm.nextInt()
                }
            }
        }
    }

    override fun random(rng: Random): Int {
        return if (rng.nextInt(size) < a.size) {
            a.random(rng)
        } else {
            b.random(rng)
        }
    }

    override fun toString() = "IntUnionCollection($a, $b)"
}
