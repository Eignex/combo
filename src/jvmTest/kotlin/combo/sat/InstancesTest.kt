package combo.sat

import combo.math.FloatVector
import combo.math.permutation
import combo.test.assertContentEquals
import combo.util.key
import combo.util.value
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.*

abstract class InstanceTest {

    abstract val factory: InstanceFactory

    @Test
    fun indices() {
        assertEquals(0 until 3, factory.create(3).indices)
    }

    @Test
    fun getDefaultFalse() {
        val instance = factory.create(10)
        for (i in instance.indices) {
            assertFalse(instance.isSet(i))
        }
    }

    @Test
    fun size() {
        for (i in 1..10) assertEquals(i, factory.create(i).size)
    }

    @Test
    fun setOne() {
        val instance = factory.create(4)
        instance[2] = true
        assertFalse(instance.isSet(0))
        assertFalse(instance.isSet(1))
        assertTrue(instance.isSet(2))
        assertFalse(instance.isSet(3))
    }

    @Test
    fun flip() {
        val instance = factory.create(5)
        for (i in instance.indices) assertFalse(instance.isSet(i))
        instance.flip(1)
        assertFalse(instance.isSet(0))
        assertTrue(instance.isSet(1))
        assertFalse(instance.isSet(2))
        assertFalse(instance.isSet(3))
        assertFalse(instance.isSet(4))
        instance.flip(1)
        for (i in instance.indices) assertFalse(instance.isSet(i))
    }

    @Test
    fun setMany() {
        val instance = factory.create(10)
        assertFalse(instance.isSet(4))
        instance[4] = true
        assertTrue(instance.isSet(4))
        instance[4] = true
        assertTrue(instance.isSet(4))
        instance[4] = false
        assertFalse(instance.isSet(4))
        instance[5] = false
        assertFalse(instance.isSet(5))
    }

    @Test
    fun setAll() {
        val instance = factory.create(3).apply { setAll(intArrayOf(1, 2, -3)) }
        assertTrue(instance.isSet(0))
        assertTrue(instance.isSet(1))
        assertFalse(instance.isSet(2))
    }

    @Test
    fun copyEquals() {
        val instance = factory.create(3).apply { setAll(intArrayOf(1, 2, -3)) }
        assertEquals(instance, instance.copy())
        assertEquals(instance.copy(), instance)
    }

    @Test
    fun copy() {
        val instance = factory.create(3).apply { setAll(intArrayOf(1, 2, -3)) }
        val instance2 = instance.copy()
        instance2[0] = false
        assertNotEquals(instance, instance2)
        assertTrue(instance.isSet(0))
        assertFalse(instance2.isSet(0))
    }

    @Test
    fun copyRandom() {
        val r = Random(1023)
        for (i in 1..50) {
            val instance = factory.create(1 + r.nextInt(1000))
            for (k in 0 until instance.size) instance[k] = r.nextBoolean()
            assertContentEquals(instance.toIntArray(), instance.copy().toIntArray())
        }
    }

    @Test
    fun empty() {
        val instance = factory.create(0)
        assertEquals(0, instance.size)
    }

    @Test
    fun equalsTransitive() {
        val instance1 = factory.create(10)
        for (k in 0 until instance1.size) instance1[k] = Random.nextBoolean()
        val instance2 = instance1.copy()
        assertEquals(instance1, instance2)
        assertEquals(instance2, instance1)
        instance2[0] = !instance2.isSet(0)
        assertNotEquals(instance1, instance2)
        assertNotEquals(instance2, instance1)
    }

    @Test
    fun equalsSize() {
        val instance1 = factory.create(10)
        val instance2 = factory.create(11)
        val instance3 = factory.create(10)
        assertNotEquals(instance1, instance2)
        assertEquals(instance1, instance3)
    }

    @Test
    fun hashCodeSize() {
        val instance1 = factory.create(10)
        val instance2 = factory.create(11)
        val instance3 = factory.create(10)
        assertNotEquals(instance1.hashCode(), instance2.hashCode())
        assertEquals(instance1.hashCode(), instance3.hashCode())
    }

    @Test
    fun equalsSet() {
        val instance1 = factory.create(20)
        val instance2 = factory.create(20)
        assertEquals(instance1, instance2)
        instance1[10] = true
        assertNotEquals(instance1, instance2)
    }

    @Test
    fun hashCodeSetDiffers() {
        val instance1 = factory.create(20)
        val instance2 = factory.create(20)
        assertEquals(instance1.hashCode(), instance2.hashCode())
        instance1[10] = true
        assertNotEquals(instance1.hashCode(), instance2.hashCode())
    }

    @Test
    fun randomHashEquals() {
        // Tests equals/hashCode implementations
        val s = generateSequence {
            factory.create(4).apply {
                for (k in 0 until 4) this[k] = Random.nextBoolean()
            }
        }.take(100).toHashSet()
        assertTrue(s.size <= 16)
    }

    @Test
    fun iteratorEmpty() {
        val instance = factory.create(10)
        assertEquals(0, instance.iterator().asSequence().count())
    }

    @Test
    fun iteratorOne() {
        val instance = factory.create(1)
        instance[0] = true
        assertContentEquals(listOf(0), instance.iterator().asSequence().toList())
        instance[0] = false
        assertContentEquals(emptyList<Int>(), instance.iterator().asSequence().toList())
    }

    @Test
    fun iterator() {
        val instance = factory.create(10)
        instance[0] = true
        instance[2] = true
        instance[9] = true
        val result = instance.iterator().asSequence().toList().toIntArray().apply { sort() }
        assertContentEquals(intArrayOf(0, 2, 9), result)
    }

    @Test
    fun iteratorLarge() {
        val instance = factory.create(100)
        for (i in instance.indices step 5)
            instance[i] = true
        val result = instance.iterator().asSequence().toList().toIntArray().apply { sort() }
        assertContentEquals((0..95 step 5).toList().toIntArray(), result)
    }

    @Test
    fun iteratorLargeHoled() {
        val instance = factory.create(200)
        instance[195] = true
        instance[85] = true
        val result = instance.iterator().asSequence().toList().toIntArray().apply { sort() }
        assertContentEquals(intArrayOf(85, 195), result)
    }

    @Test
    fun iteratorLastElement() {
        val instance = factory.create(99)
        instance[31] = true
        instance[98] = true
        val result = instance.iterator().asSequence().toList().toIntArray().apply { sort() }
        assertContentEquals(intArrayOf(31, 98), result)
    }

    @Test
    fun iteratorAllOnes() {
        val instance = factory.create(227)
        for (i in instance.indices) instance[i] = true
        val result = instance.iterator().asSequence().toList().toIntArray().apply { sort() }
        assertContentEquals((0 until 227).toList().toIntArray(), result)
    }

    @Test
    fun flipLarge() {
        val instance = factory.create(101)
        for (i in instance.indices) assertFalse(instance.isSet(i))
        for (i in instance.indices step 3) instance.flip(i)
        for (i in instance.indices) if (i.rem(3) == 0) assertTrue(instance.isSet(i)) else assertFalse(instance.isSet(i))
        for (i in instance.indices step 3) instance.flip(i)
        for (i in instance.indices) assertFalse(instance.isSet(i))
    }

    @Test
    fun flipRandom() {
        for (i in 1..10) {
            val r = Random.Default
            val instance = factory.create(1 + r.nextInt(1001))
            for (j in 1..100) {
                val id = r.nextInt(instance.size)
                val b1 = instance.isSet(id)
                instance.flip(id)
                assertNotEquals(b1, instance.isSet(id))
            }
        }
    }

    @Test
    fun bitsSetGetBounds() {
        val instance = factory.create(100)
        instance.setBits(50, 32, Int.MIN_VALUE)
        assertEquals(Int.MIN_VALUE, instance.getBits(50, 32))

        instance.setBits(42, 32, -1)
        assertEquals(-1, instance.getBits(42, 32))

        instance.setBits(20, 32, Int.MAX_VALUE)
        assertEquals(Int.MAX_VALUE, instance.getBits(20, 32))
    }

    @Test
    fun bitsSetGetBoundsByte() {
        val instance = factory.create(100)
        instance.setBits(10, 8, 0b10000000)
        assertEquals(Byte.MIN_VALUE, instance.getBits(10, 8).toByte())

        instance.setBits(40, 8, Byte.MAX_VALUE.toInt())
        assertEquals(Byte.MAX_VALUE, instance.getBits(40, 8).toByte())
    }

    @Test
    fun bitsSetGetArbitrary() {
        val instance = factory.create(13)
        instance.setBits(0, 5, 1)
        assertEquals(1, instance.getBits(0, 5))

        instance.setBits(5, 8, 255)
        assertEquals(255, instance.getBits(5, 8))
    }

    @Test
    fun setBitsTmp() {
        val instance = factory.create(64)
        instance.setBits(30, 4, 5)
    }

    @Test
    fun bitsSetGetManyInts() {
        val instance = factory.create(200)
        for (i in 0..168) {
            val value = Random.nextInt()
            instance.setBits(i, 32, value)
            assertEquals(value, instance.getBits(i, 32))
        }
    }

    @Test
    fun bitsSetGetManyShorts() {
        val instance = factory.create(200)
        val rng = Random(0)
        for (i in 0..184) {
            for (j in instance.indices) instance[j] = false
            val value = rng.nextInt(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
            instance.setSignedInt(i, 16, value)
            assertEquals(value, instance.getSignedInt(i, 16), "$i")
        }
    }

    @Test
    fun getSetFloat() {
        val instance = factory.create(32)
        for (i in 1..10) {
            val rnd = Float.fromBits(Random.nextBits(32))
            instance.setFloat(0, rnd)
            assertEquals(rnd, instance.getFloat(0))
        }
    }

    @Test
    fun bitsIsolated() {
        val instance = factory.create(65)
        for (i in instance.indices) instance[i] = true
        instance.setBits(63, 2, 1)
        for (i in 0 until 63) assertTrue(instance.isSet(i))
        assertEquals(1, instance.getBits(63, 2))

        instance.setBits(50, 15, 32767)
        for (i in 0 until 63) assertTrue(instance.isSet(i))
        instance.setBits(50, 15, 0)
        for (i in 0 until 50) assertTrue(instance.isSet(i))
        for (i in 50 until 65) assertFalse(instance.isSet(i))
    }

    @Test
    fun bitsIsolatedEven() {
        val instance = factory.create(96)
        instance.setBits(0, 32, -1)
        instance.setBits(32, 32, 0)
        assertEquals(-1, instance.getBits(0, 32))
        instance.setBits(64, 32, -1)
        assertEquals(0, instance.getBits(32, 32))
    }

    @Test
    fun bitsIsolatedUneven() {
        val instance = factory.create(118)
        instance.setBits(54, 32, -1)
        instance.setBits(22, 32, 0)
        instance.setBits(86, 32, 0)
        assertEquals(-1, instance.getBits(54, 32))
        instance.setBits(54, 32, -1)
        assertEquals(0, instance.getBits(22, 32))
        assertEquals(0, instance.getBits(86, 32))

        instance.setBits(54, 32, 0)
        instance.setBits(22, 32, -1)
        instance.setBits(86, 32, -1)
        assertEquals(0, instance.getBits(54, 32))
        assertEquals(-1, instance.getBits(22, 32))
        assertEquals(-1, instance.getBits(86, 32))
    }

    @Test
    fun getSetSignedBits10() {
        val instance = factory.create(10)

        instance.setSignedInt(0, 10, 8)
        assertEquals(8, instance.getSignedInt(0, 10))

        instance.setSignedInt(0, 10, -2)
        assertEquals(-2, instance.getSignedInt(0, 10))
    }

    @Test
    fun getSetSignedBits32() {
        val instance = factory.create(32)

        instance.setSignedInt(0, 32, Int.MIN_VALUE)
        assertEquals(Int.MIN_VALUE, instance.getSignedInt(0, 32))

        instance.setSignedInt(0, 32, Int.MIN_VALUE)
        assertEquals(Int.MIN_VALUE, instance.getSignedInt(0, 32))

        instance.setSignedInt(0, 32, -1)
        assertEquals(-1, instance.getSignedInt(0, 32))
    }

    @Test
    fun wordSize() {
        assertEquals(2, factory.create(60).wordSize)
        assertEquals(1, factory.create(32).wordSize)
        assertEquals(0, factory.create(0).wordSize)
        assertEquals(2, factory.create(64).wordSize)
        assertEquals(3, factory.create(65).wordSize)
    }

    @Test
    fun wordIterator() {
        val instance1 = factory.create(100)
        instance1[10] = true
        instance1[50] = true
        val list = instance1.wordIterator().asSequence().toList().sortedBy { it.key() }
        assertEquals(1024 /*2^10*/, list[0].value())
        assertEquals(262144 /*2^18=2^(50-32)*/, list[1].value())
        assertEquals(0, list[0].key())
        assertEquals(1, list[1].key())
    }

    @Test
    fun orInstance() {
        val inst1 = factory.create(60)
        val inst2 = factory.create(60)
        inst1[10] = true
        inst1[30] = true
        inst2[20] = true
        val inst3 = inst1.copy().apply { or(inst2) }
        val inst4 = inst2.copy().apply { or(inst1) }
        assertEquals(inst3, inst4)

        assertTrue(inst3.isSet(10))
        assertTrue(inst3.isSet(20))
        assertTrue(inst3.isSet(30))

        assertTrue(inst4.isSet(10))
        assertTrue(inst4.isSet(20))
        assertTrue(inst4.isSet(30))
        assertEquals(3, inst4.iterator().asSequence().count())
    }

    @Test
    fun andInstance() {
        val inst1 = factory.create(60)
        val inst2 = factory.create(60)
        inst1[10] = true
        inst1[30] = true
        inst1[40] = true
        inst1[60] = true
        inst2[10] = true
        inst2[20] = true
        inst2[30] = true
        val inst3 = inst1.copy().apply { and(inst2) }
        val inst4 = inst2.copy().apply { and(inst1) }

        assertEquals(inst3, inst4)

        assertTrue(inst3.isSet(10))
        assertFalse(inst3.isSet(20))
        assertTrue(inst3.isSet(30))
        assertFalse(inst3.isSet(40))
        assertFalse(inst3.isSet(60))

        assertEquals(2, inst4.iterator().asSequence().count())
    }

    @Test
    fun dot() {
        val inst = factory.create(100)
        val vec = FloatVector(FloatArray(100) { it.toFloat() })
        assertEquals(0.0f, inst.dot(vec))
        assertEquals(vec dot inst, inst dot vec)
        inst[2] = true
        assertEquals(2.0f, inst dot vec)
        assertEquals(vec dot inst, inst dot vec)
        inst[90] = true
        assertEquals(92.0f, inst dot vec)
        assertEquals(vec dot inst, inst dot vec)
    }

    @Test
    fun norm2() {
        val inst = factory.create(10)
        assertEquals(0f, inst.norm2())
        inst[1] = true
        inst[3] = true
        assertEquals(sqrt(2f), inst.norm2())
    }

    @Test
    fun plus() {
        val inst = factory.create(3)
        inst[1] = true
        val vec = FloatVector(floatArrayOf(1f, 2f, 3.1f))
        val u1 = inst + vec
        val u2 = vec + inst
        assertContentEquals(floatArrayOf(1f, 3f, 3.1f), u1.toFloatArray())
        assertContentEquals(floatArrayOf(1f, 3f, 3.1f), u2.toFloatArray())
        val u3 = inst + 1.1f
        assertContentEquals(floatArrayOf(1.1f, 2.1f, 1.1f), u3.toFloatArray())
    }

    @Test
    fun minus() {
        val inst = factory.create(3)
        inst[1] = true
        val vec = FloatVector(floatArrayOf(1f, 2f, 3.1f))
        val u1 = inst - vec
        val u2 = vec - inst
        assertContentEquals(floatArrayOf(-1f, -1f, -3.1f), u1.toFloatArray())
        assertContentEquals(floatArrayOf(1f, 1f, 3.1f), u2.toFloatArray())
        val u3 = inst - 2f
        assertContentEquals(floatArrayOf(-2.0f, -1.0f, -2.0f), u3.toFloatArray())
    }

    @Test
    fun times() {
        val inst = factory.create(3)
        inst[1] = true
        val vec = FloatVector(floatArrayOf(1f, 2f, 3.1f))
        val u1 = inst * vec
        val u2 = vec * inst
        assertContentEquals(floatArrayOf(0f, 2f, 0f), u1.toFloatArray())
        assertContentEquals(floatArrayOf(0f, 2f, 0f), u2.toFloatArray())
        val u3 = inst * 2f
        assertContentEquals(floatArrayOf(0.0f, 2.0f, 0.0f), u3.toFloatArray())
    }

    @Test
    fun divide() {
        val inst = factory.create(3)
        inst[1] = true
        val vec = FloatVector(floatArrayOf(1f, 2f, 3.1f))
        val u1 = inst / vec
        assertContentEquals(floatArrayOf(0f, 0.5f, 0f), u1.toFloatArray())
        val u3 = inst / 2f
        assertContentEquals(floatArrayOf(0f, 0.5f, 0f), u3.toFloatArray())
        inst[0] = true; inst[2] = true
        val u2 = vec / inst
        assertContentEquals(floatArrayOf(1f, 2f, 3.1f), u2.toFloatArray())
    }

    @Test
    fun cardinality() {
        val instance = factory.create(1)
        instance[0] = true
        assertEquals(1, instance.cardinality())
        instance[0] = false
        assertEquals(0, instance.cardinality())
    }

    @Test
    fun cardinalityMultiWords() {
        val instance = factory.create(100)
        assertEquals(0, instance.cardinality())
        instance[0] = true
        instance[80] = true
        assertEquals(2, instance.cardinality())
        for (i in 0 until 100)
            instance[i] = true
        assertEquals(100, instance.cardinality())
    }

    @Test
    fun getFirst() {
        val inst = factory.create(60)
        assertEquals(-1, inst.getFirst(0, 60))
        inst[10] = true
        assertEquals(10, inst.getFirst(0, 60))
        assertEquals(10, inst.getFirst(0, 11))
        assertEquals(-1, inst.getFirst(0, 10))
        assertEquals(-1, inst.getFirst(11, 60))
        assertEquals(0, inst.getFirst(10, 60))
        inst[50] = true
        assertEquals(10, inst.getFirst(0, 60))
    }

    @Test
    fun getFirstSingular() {
        val inst = factory.create(1)
        assertEquals(-1, inst.getFirst(0, 1))
        inst[0] = true
        assertEquals(0, inst.getFirst(0, 1))
    }

    @Test
    fun getFirstBig() {
        val inst = factory.create(200)
        inst[199] = true
        assertEquals(199, inst.getFirst(0, 200))
    }

    @Test
    fun getFistEvenWord() {
        val inst1 = factory.create(32)
        inst1[0] = true
        assertEquals(0, inst1.getFirst(0, 1))
        inst1[0] = false
        inst1[31] = true
        assertEquals(31, inst1.getFirst(0, 32))

        val inst2 = factory.create(128)
        inst2[0] = true
        assertEquals(0, inst2.getFirst(0, 1))
        inst2[0] = false
        inst2[127] = true
        assertEquals(63, inst2.getFirst(64, 128))
    }

    @Test
    fun getLast() {
        val inst = factory.create(60)
        assertEquals(-1, inst.getLast(0, 60))
        inst[10] = true
        assertEquals(10, inst.getLast(0, 60))
        assertEquals(10, inst.getLast(0, 11))
        assertEquals(-1, inst.getLast(0, 10))
        assertEquals(-1, inst.getLast(11, 60))
        assertEquals(0, inst.getLast(10, 60))
        inst[50] = true
        assertEquals(50, inst.getLast(0, 60))
    }

    @Test
    fun getLastBig() {
        val inst = factory.create(200)
        inst[199] = true
        assertEquals(199, inst.getLast(0, 200))
    }

    @Test
    fun getLastEvenWord() {
        val inst1 = factory.create(32)
        inst1[0] = true
        assertEquals(0, inst1.getLast(0, 1))
        inst1[0] = false
        inst1[31] = true
        assertEquals(31, inst1.getLast(0, 32))

        val inst2 = factory.create(128)
        inst2[0] = true
        assertEquals(0, inst2.getLast(0, 1))
        inst2[0] = false
        inst2[127] = true
        assertEquals(63, inst2.getLast(64, 128))
    }


    @Test
    fun hashCodeOrder() {
        val rng = Random
        val log2Size = 11
        val baseInstance = factory.create(2.0.pow(log2Size).toInt())
        val wordSize = baseInstance.wordSize
        val words = IntArray(wordSize) { rng.nextInt() }
        for (wi in 0 until wordSize)
            baseInstance.setWord(wi, words[wi])
        val baseHashCode = baseInstance.hashCode()
        for (j in 0 until 10) {
            val permutedInstance = factory.create(baseInstance.size)
            for (wi in permutation(wordSize, rng)) {
                if (permutedInstance.getWord(wi) != words[wi])
                    permutedInstance.setWord(wi, words[wi])
            }
            for (k in permutation(baseInstance.size * 5, rng)) {
                val wi = k % wordSize
                if (rng.nextBoolean()) permutedInstance.setWord(wi, 0)
                else if (rng.nextBoolean()) permutedInstance.setWord(wi, rng.nextInt())
                else permutedInstance.setWord(wi, words[wi])
            }
            for (wi in permutation(wordSize, rng)) {
                if (permutedInstance.getWord(wi) != words[wi])
                    permutedInstance.setWord(wi, words[wi])
            }
            val permutedHashCode = permutedInstance.hashCode()
            assertEquals(baseHashCode, permutedHashCode)
        }
    }
}

class SparseBitArrayTest : InstanceTest() {
    override val factory = SparseBitArrayFactory
}

class BitArrayTest : InstanceTest() {
    override val factory = BitArrayFactory
}

