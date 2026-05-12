package combo.util

import java.util.concurrent.atomic.AtomicInteger
import kotlin.random.Random

/**
 * This class is for having thread safe Random objects that is guaranteed to be deterministic in a non-threaded
 * environment (like unit tests). This works very similar to seed sequence in C++.
 */
class RandomSequence(val sequenceStart: Int) {
    private val seedUniquifier = AtomicInteger(sequenceStart)
    fun next(): Random {
        var current: Int
        var next: Int
        do {
            current = seedUniquifier.get()
            next = (1 + current) * 741103587 // This number comes from the linear congruent generator paper
        } while (!seedUniquifier.compareAndSet(current, next))
        return Random(next)
    }
}
