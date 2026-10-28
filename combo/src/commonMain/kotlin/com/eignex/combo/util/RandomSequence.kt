@file:OptIn(kotlin.concurrent.atomics.ExperimentalAtomicApi::class)

package com.eignex.combo.util

import kotlin.concurrent.atomics.AtomicInt
import kotlin.random.Random

/**
 * This class is for having thread safe Random objects that is guaranteed to be deterministic in a non-threaded
 * environment (like unit tests). This works very similar to seed sequence in C++.
 */
class RandomSequence(val sequenceStart: Int) {
    private val seedUniquifier = AtomicInt(sequenceStart)
    fun next(): Random {
        var current: Int
        var next: Int
        do {
            current = seedUniquifier.load()
            next = (1 + current) * 741103587 // This number comes from the linear congruent generator paper
        } while (!seedUniquifier.compareAndSet(current, next))
        return Random(next)
    }
}
