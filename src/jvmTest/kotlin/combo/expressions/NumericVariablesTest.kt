package combo.expressions

import combo.model.VariableIndex
import combo.sat.*
import combo.test.assertContentEquals
import combo.util.*
import kotlin.random.Random
import kotlin.test.*

class IntVarTest {

    @Test
    fun minEqMax() {
        assertFailsWith(IllegalArgumentException::class) {
            IntVar("", true, Root(""), 0, 0)
        }
    }

    @Test
    fun literalSize() {
        assertEquals(3, IntVar("", false, Root(""), -4, 0).nbrValues)
        assertEquals(3, IntVar("", false, Root(""), -4, 3).nbrValues)
        assertEquals(3, IntVar("", true, Root(""), 0, 3).nbrValues)
        assertEquals(2, IntVar("", false, Root(""), 0, 2).nbrValues)

        assertEquals(6, IntVar("", true, Root(""), -10, 8).nbrValues)
        assertEquals(5, IntVar("", false, Root(""), -16, 15).nbrValues)
        assertEquals(5, IntVar("", false, Root(""), -10, 15).nbrValues)
    }

    @Test
    fun literalSizeBounds() {
        fun limits(v: IntVar) {
            val min = 1
            val max = 32 + if (v.optional) 1 else 0
            assertTrue(v.nbrValues in min..max)
        }
        limits(IntVar("", false, Root(""), Int.MIN_VALUE, Int.MAX_VALUE))
        limits(IntVar("", false, Root(""), Int.MIN_VALUE, 0))
        limits(IntVar("", false, Root(""), 0, Int.MAX_VALUE))
        limits(IntVar("", false, Root(""), -1, Int.MAX_VALUE))
        limits(IntVar("", true, Root(""), Int.MIN_VALUE, Int.MAX_VALUE))
        limits(IntVar("", true, Root(""), Int.MIN_VALUE, 0))
        limits(IntVar("", true, Root(""), 0, Int.MAX_VALUE))
        limits(IntVar("", true, Root(""), -1, Int.MAX_VALUE))
    }

    @Test
    fun indexOptional() {
        val f = IntVar("", true, Root(""), min = 0, max = 1)
        val index = VariableIndex()
        index.add(f)
        assertEquals(0, index.valueIndexOf(f))
        assertEquals(2, f.nbrValues)
    }

    @Test
    fun indexMandatory() {
        val f = IntVar("", false, Root(""), min = 0, max = 1)
        val index = VariableIndex()
        index.add(f)
        assertEquals(0, index.valueIndexOf(f))
        assertEquals(1, f.nbrValues)
    }

    @Test
    fun valueOfOptional() {
        val f = IntVar("", true, Root(""), min = -1, max = 10)
        val instance = BitArray(f.nbrValues)
        assertNull(f.valueOf(instance, 0, 0))
        instance[0] = true
        assertEquals(0, f.valueOf(instance, 0, 0))
        instance.setBits(1, f.nbrValues - 1, 4)
        assertEquals(4, f.valueOf(instance, 0, 0))
        for (i in instance.indices) instance[i] = true
        val actual = f.valueOf(instance, 0, 0)
        assertEquals(-1, actual)
    }

    @Test
    fun valueOfMandatory() {
        val f = IntVar("", false, Root(""), min = -1, max = 10)
        val index = VariableIndex()
        index.add(f)
        val instance = BitArray(f.nbrValues)
        assertEquals(0, f.valueOf(instance, 0, 0))
        instance.setBits(0, f.nbrValues, 4)
        assertEquals(4, f.valueOf(instance, 0, 0))
    }

    @Test
    fun valueOfMandatoryNonZero() {
        val f = IntVar("", false, Root(""), min = 1, max = 10)
        val index = VariableIndex()
        index.add(f)
        val instance = BitArray(f.nbrValues)
        instance.setBits(0, f.nbrValues, 4)
        assertEquals(4, f.valueOf(instance, 0, 0))
    }

    @Test
    fun toLiteralMandatory() {
        val p = Flag("p", true, Root(""))
        val f = IntVar("", false, p, min = -1000, max = 100)
        val index = VariableIndex()
        index.add(p)
        index.add(f)
        assertEquals(1, f.toLiteral(index))
    }

    @Test
    fun toLiteralOptional() {
        val f = IntVar("", true, Root(""), min = -12489121, max = -41)
        val index = VariableIndex()
        index.add(BitsVar("b", false, Root(""), 5))
        index.add(f)
        assertEquals(6, f.toLiteral(index))
    }
}

class IntLiteralTest {

    @Test
    fun outOfBounds() {
        val v = IntVar("v", false, Root(""), 1, 20)
        assertFailsWith(IllegalArgumentException::class) {
            v.value(21)
        }
    }

    @Test
    fun toLiteralOptional() {
        fun testToLiteral(min: Int, max: Int, value: Int) {
            val v = IntVar("v", true, Root(""), min, max)
            val index = VariableIndex()
            index.add(v)
            val instance = BitArray(index.nbrValues)
            val set = IntHashSet()
            v.value(value).collectLiterals(index, set)
            Conjunction(set).coerce(instance, Random)
            assertEquals(value, v.valueOf(instance, index.valueIndexOf(v), 0))
        }

        for (i in -10..10) testToLiteral(-10, 10, i)
        for (i in 1..10) testToLiteral(1, 10, i)
        for (i in -1 downTo -10) testToLiteral(-10, -1, i)
        testToLiteral(Int.MIN_VALUE, Int.MAX_VALUE, -1)
        testToLiteral(Int.MIN_VALUE, Int.MAX_VALUE, Int.MIN_VALUE)
        testToLiteral(Int.MIN_VALUE, Int.MAX_VALUE, Int.MAX_VALUE)
    }

    @Test
    fun toLiteralMandatory() {
        val a = IntVar("a", false, Root(""), 0, 5)
        val index = VariableIndex()
        index.add(a)
        fun test(literals: IntArray, value: Int) {
            val set = IntHashSet()
            a.value(value).collectLiterals(index, set)
            val arr = set.toArray().apply { sort() }
            assertContentEquals(literals.apply { sort() }, arr)
        }
        test(intArrayOf(-1, -2, -3), 0)
        test(intArrayOf(1, -2, -3), 1)
        test(intArrayOf(-1, 2, -3), 2)
        test(intArrayOf(1, 2, -3), 3)
        test(intArrayOf(-1, -2, 3), 4)
        test(intArrayOf(1, -2, 3), 5)
    }
}

class FloatVarTest {

    @Test
    fun minEqMax() {
        assertFailsWith(IllegalArgumentException::class) {
            FloatVar("", true, Root(""), 0f, 0f)
        }
    }

    @Test
    fun indexOptional() {
        val f = FloatVar("", true, Root(""), 0f, 1f)
        val index = VariableIndex()
        index.add(f)
        assertEquals(0, index.valueIndexOf(f))
        assertEquals(33, f.nbrValues)
    }

    @Test
    fun indexMandatory() {
        val f = FloatVar("", false, Root(""), 0f, 1f)
        val index = VariableIndex()
        index.add(f)
        assertEquals(0, index.valueIndexOf(f))
        assertEquals(32, f.nbrValues)
    }

    @Test
    fun valueOfOptional() {
        val f = FloatVar("", true, Root(""), -1f, 10f)
        val instance = BitArray(f.nbrValues)
        assertNull(f.valueOf(instance, 0, 0))
        instance[0] = true
        assertEquals(0f, f.valueOf(instance, 0, 0))
        instance.setFloat(1, 4.5f)
        assertEquals(4.5f, f.valueOf(instance, 0, 0))
    }

    @Test
    fun valueOfMandatory() {
        val f = FloatVar("", false, Root(""), -1f, 10f)
        val instance = BitArray(f.nbrValues)
        assertEquals(0f, f.valueOf(instance, 0, 0))
        instance.setFloat(0, 4.5f)
        assertEquals(4.5f, f.valueOf(instance, 0, 0))
    }

    @Test
    fun valueOfMandatoryNonZero() {
        val f = FloatVar("", false, Root(""), 1f, 10f)
        val instance = BitArray(f.nbrValues)
        instance.setFloat(0, 4f)
        assertEquals(4f, f.valueOf(instance, 0, 0))
    }

    @Test
    fun toLiteralOptional() {
        val parent = Flag("p", true, Root(""))
        val f = FloatVar("", true, parent, -1000.1f, 1000.0f)
        val index = VariableIndex()
        index.add(parent)
        index.add(f)
        assertEquals(2, f.toLiteral(index))
    }

    @Test
    fun toLiteralMandatory() {
        val parent = Flag("p", true, Root(""))
        val f = FloatVar("", false, parent, 1f, 10f)
        val index = VariableIndex()
        index.add(BitsVar("b", true, Root(""), 5))
        index.add(parent)
        index.add(f)
        assertEquals(7, f.toLiteral(index))
    }
}

class FloatLiteralTest {

    @Test
    fun outOfBounds() {
        val v = FloatVar("v", true, Root(""), 1f, 20f)
        assertFailsWith(IllegalArgumentException::class) {
            v.value(21f)
        }
    }

    @Test
    fun nonFinite() {
        val v = FloatVar("v", false, Root(""), 1f, 20f)
        assertFailsWith(IllegalArgumentException::class) {
            v.value(Float.NaN)
        }
        assertFailsWith(IllegalArgumentException::class) {
            v.value(Float.POSITIVE_INFINITY)
        }
        assertFailsWith(IllegalArgumentException::class) {
            v.value(Float.NEGATIVE_INFINITY)
        }
    }

    @Test
    fun toLiteralOptional() {
        fun testToLiteral(min: Float, max: Float, value: Float) {
            val v = FloatVar("v", true, Root(""), min, max)
            val index = VariableIndex()
            index.add(v)
            val instance = BitArray(index.nbrValues)
            val set = IntHashSet()
            v.value(value).collectLiterals(index, set)
            Conjunction(set).coerce(instance, Random)
            assertEquals(value.toBits(), v.valueOf(instance, 0, 0)!!.toBits())
        }

        for (i in -10..10) testToLiteral(-10F, 10F, i.toFloat())
        for (i in 1..10) testToLiteral(0F, 1F, 1F / i.toFloat())
        for (i in -1 downTo -10) testToLiteral(-10F, -1F, i.toFloat())
        testToLiteral(-MAX_VALUE32, MAX_VALUE32, 0F)
        testToLiteral(-MAX_VALUE32, MAX_VALUE32, MAX_VALUE32)
        testToLiteral(-MAX_VALUE32, MAX_VALUE32, -MAX_VALUE32)
        testToLiteral(-MAX_VALUE32, MAX_VALUE32, MIN_VALUE32)
        testToLiteral(-MAX_VALUE32, MAX_VALUE32, -MIN_VALUE32)
    }

    @Test
    fun toLiteralMandatory() {
        fun testToLiteral(min: Float, max: Float, value: Float) {
            val v = FloatVar("v", false, Root(""), min, max)
            val index = VariableIndex()
            index.add(v)
            val instance = BitArray(index.nbrValues)
            val set = IntHashSet()
            v.value(value).collectLiterals(index, set)
            Conjunction(set).coerce(instance, Random)
            assertEquals(value.toBits(), v.valueOf(instance, 0, 0)!!.toBits())
        }

        for (i in -10..10) testToLiteral(-10F, 10F, i.toFloat())
        for (i in 1..10) testToLiteral(0F, 1F, 1F / i.toFloat())
        for (i in -1 downTo -10) testToLiteral(-10F, -1F, i.toFloat())
        testToLiteral(-MAX_VALUE32, MAX_VALUE32, 0F)
        testToLiteral(-MAX_VALUE32, MAX_VALUE32, MAX_VALUE32)
        testToLiteral(-MAX_VALUE32, MAX_VALUE32, -MAX_VALUE32)
        testToLiteral(-MAX_VALUE32, MAX_VALUE32, MIN_VALUE32)
        testToLiteral(-MAX_VALUE32, MAX_VALUE32, -MIN_VALUE32)
    }
}

class BitsVarTest {

    @Test
    fun noBits() {
        assertFailsWith(IllegalArgumentException::class) {
            BitsVar("", false, Root(""), 0)
        }
    }

    @Test
    fun indexOptional() {
        val f = BitsVar("", true, Root(""), 1)
        val index = VariableIndex()
        index.add(f)
        assertEquals(0, index.valueIndexOf(f))
        assertEquals(2, f.nbrValues)
    }

    @Test
    fun indexMandatory() {
        val f = BitsVar("", false, Root(""), 1)
        val index = VariableIndex()
        index.add(f)
        assertEquals(0, index.valueIndexOf(f))
        assertEquals(1, f.nbrValues)
    }

    @Test
    fun valueOfOptional() {
        val f = BitsVar("", true, Root(""), 10)
        val instance = BitArray(f.nbrValues)
        assertNull(f.valueOf(instance, 0, 0))
        instance[0] = true
        assertEquals(BitArray(10), f.valueOf(instance, 0, 0))
        instance[1] = true
        assertEquals(
            BitArray(10).apply { this[0] = true },
            f.valueOf(instance, 0, 0)
        )
        instance[4] = true
        assertEquals(
            BitArray(10).apply { this[0] = true; this[3] = true },
            f.valueOf(instance, 0, 0)
        )
    }

    @Test
    fun valueOfMandatory() {
        val f = BitsVar("", false, Root(""), 10)
        val instance = BitArray(f.nbrValues)
        assertEquals(BitArray(10), f.valueOf(instance, 0, 0))
        instance[0] = true
        assertEquals(
            BitArray(10).apply { this[0] = true },
            f.valueOf(instance, 0, 0)
        )
        instance[3] = true
        assertEquals(
            BitArray(10).apply { this[0] = true; this[3] = true },
            f.valueOf(instance, 0, 0)
        )
    }

    @Test
    fun toLiteral() {
        val f = BitsVar("", true, Root(""), 100)
        val index = VariableIndex()
        index.add(f)
        assertEquals(1, f.toLiteral(index))
    }

    @Test
    fun toLiteralMandatory() {
        val parent = Flag("p", 1, Root(""))
        val f = BitsVar("", false, parent, 100)
        val index = VariableIndex()
        index.add(parent)
        index.add(f)
        assertEquals(1, f.toLiteral(index))
    }
}

class BitValueTest {

    @Test
    fun outOfBounds() {
        val b = BitsVar("b", true, Root(""), 10)
        assertFailsWith(IllegalArgumentException::class) {
            b.value(10)
        }
    }

    @Test
    fun toLiteralOptional() {
        val b = BitsVar("b", true, Root(""), 5)
        val index = VariableIndex()
        index.add(b)
        assertEquals(3, b.value(1).toLiteral(index))
    }

    @Test
    fun toLiteralMandatory() {
        val parent = Root("")
        val b = BitsVar("b", false, parent, 5)
        val index = VariableIndex()
        index.add(b)
        assertEquals(2, b.value(1).toLiteral(index))
    }

    @Test
    fun toLiteralNot() {
        val b = BitsVar("b", true, Root(""), 10)
        val v = b.value(2).not()
        val index = VariableIndex()
        index.add(b)
        assertEquals(b, v.canonicalVariable)
        assertEquals(1, b.toLiteral(index))
        assertEquals(2, b.value(0).toLiteral(index))
        assertEquals(-4, v.toLiteral(index))
    }

    @Test
    fun toLiteralMandatoryNot() {
        val b = BitsVar("b", false, Root(""), 10)
        val v = b.value(9).not()
        val index = VariableIndex()
        index.add(b)
        assertEquals(b, v.canonicalVariable)
        assertEquals(-10, v.toLiteral(index))
    }
}
