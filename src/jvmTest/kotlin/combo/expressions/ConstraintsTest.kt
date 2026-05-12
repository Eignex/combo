package combo.expressions

import combo.math.permutation
import combo.sat.*
import combo.util.collectionOf
import kotlin.random.Random
import kotlin.test.*

abstract class ConstraintTest {
    fun randomExhaustivePropagations(constraint: Constraint) {
        // This test thoroughly test that unit propagation does not change the truth value of an instance.
        // It iteratively calls unitPropagation on each literal in the instance.
        val rng = Random.Default
        for (i in constraint.literals) require(i.toIx() <= 4)
        for (instance in InstancePermutation(5, BitArrayFactory, rng)) {
            val c2 = permutation(instance.size, rng).iterator().asSequence()
                .fold(constraint) { s: Constraint, i ->
                    val v = instance.isSet(i)
                    val cp = s.unitPropagation(i.toLiteral(v))
                    if (cp.isUnit()) {
                        val expected = constraint.satisfies(instance)
                        val actual =
                            Conjunction(collectionOf(*cp.unitLiterals())).satisfies(
                                instance
                            )
                        assertEquals(expected, actual)
                    }
                    assertEquals(
                        constraint.satisfies(instance),
                        cp.satisfies(instance)
                    )
                    cp
                }
            assertTrue(
                c2.isUnit() || c2 == Tautology || c2 == Empty,
                "$constraint + ${instance.toLiterals().joinToString()} -> $c2"
            )
            if (c2 == Empty) assertFalse(constraint.satisfies(instance))
        }
    }

    fun randomCacheUpdates(instance: Instance, constraint: Constraint) {
        val preCache = constraint.cache(instance)
        assertEquals(
            constraint.violations(instance),
            constraint.violations(instance, preCache),
            "$preCache: $instance"
        )
        val lit = constraint.literals.random(Random)
        val updatedCache =
            constraint.cacheUpdate(preCache, !instance.literal(lit.toIx()))
        instance.flip(lit.toIx())
        assertEquals(constraint.cache(instance), updatedCache)
        assertEquals(
            constraint.violations(instance),
            constraint.violations(instance, updatedCache)
        )
    }

    fun randomCoerce(constraint: Constraint): List<Instance> {
        val rng = Random
        for (i in constraint.literals) require(i.toIx() <= 4)
        val list = ArrayList<Instance>()
        for (instance in InstancePermutation(5, BitArrayFactory, rng)) {
            constraint.coerce(instance, rng)
            assertTrue(constraint.satisfies(instance))
            list.add(instance)
        }
        return list
    }
}
