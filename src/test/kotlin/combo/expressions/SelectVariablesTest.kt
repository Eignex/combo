package combo.expressions


import combo.model.VariableIndex
import combo.sat.BitArray
import combo.test.assertContentEquals
import kotlin.test.*

class NominalTest {

    @Test
    fun indexOptional() {
        val n = Nominal("n", true, Root(""), 1, 2, 3)
        val index = VariableIndex()
        index.add(n)
        assertEquals(0, index.valueIndexOf(n))
        assertEquals(4, n.nbrValues)
    }

    @Test
    fun indexMandatory() {
        val n = Nominal("n", false, Root(""), 1, 2, 3)
        val index = VariableIndex()
        index.add(n)
        assertEquals(0, index.valueIndexOf(n))
        assertEquals(3, n.nbrValues)
    }

    @Test
    fun valueOfOptional() {
        val parent = Flag("f", 1, Root(""))
        val n = Nominal("n", true, parent, "a", "b", "c", "d")
        val instance = BitArray(6)
        instance[0] = true
        assertNull(n.valueOf(instance, 1, 1))
        instance[1] = true
        assertFailsWith(IllegalStateException::class) {
            n.valueOf(instance, 1, 1)
        }
        instance[3] = true
        assertEquals("b", n.valueOf(instance, 1, 1))
    }

    @Test
    fun valueOfMandatory() {
        val n = Nominal("n", false, Root(""), "a", "b", "c", "d")
        val instance = BitArray(4)
        assertFailsWith<IllegalStateException> {
            assertNull(n.valueOf(instance, 0, 0))
        }
        instance[1] = true
        assertEquals("b", n.valueOf(instance, 0, 0))
    }

    @Test
    fun toLiteralOptional() {
        val n = Nominal("n", true, Root(""), "a", "b", "c")
        val index = VariableIndex()
        index.add(n)
        assertEquals(1, n.toLiteral(index))
    }

    @Test
    fun toLiteralMandatory() {
        val n = Nominal("n", false, Root(""), "a", "b", "c")
        val index = VariableIndex()
        index.add(n)
        assertFailsWith(Exception::class) { n.toLiteral(index) }
    }

    @Test
    fun toLiteral2() {
        val root = Root("root")
        val parent = Flag("p", true, root)
        val n = Nominal("n", true, parent, null, 1, 2, 3, 4, 5, 6, 7)
        val index = VariableIndex()
        index.add(parent)
        index.add(BitsVar("b", false, root, 5))
        index.add(n)
        assertEquals(7, n.toLiteral(index))
    }
}

class MultipleTest {

    @Test
    fun indexOptional() {
        val a = Multiple("a", true, Root(""), 1, 2, 3)
        val index = VariableIndex()
        index.add(a)
        assertEquals(0, index.valueIndexOf(a))
        assertEquals(4, a.nbrValues)
    }

    @Test
    fun indexMandatory() {
        val a = Multiple("a", false, Root(""), 1, 2, 3)
        val index = VariableIndex()
        index.add(a)
        assertEquals(0, index.valueIndexOf(a))
        assertEquals(3, a.nbrValues)
    }

    @Test
    fun valueOfOptional() {
        val a = Multiple("a", true, Root(""), "a", "b", "c", "d")
        val instance = BitArray(5)
        assertNull(a.valueOf(instance, 0, 0))
        instance[0] = true
        assertFailsWith(IllegalStateException::class) {
            a.valueOf(instance, 0, 0)
        }
        instance[2] = true
        assertContentEquals(listOf("b"), a.valueOf(instance, 0, 0)!!)
        instance[1] = true
        assertContentEquals(listOf("a", "b"), a.valueOf(instance, 0, 0)!!)
        instance[3] = true
        assertContentEquals(listOf("a", "b", "c"), a.valueOf(instance, 0, 0)!!)
    }

    @Test
    fun valueOfMandatory() {
        val a = Multiple("a", false, Root(""), "a", "b", "c", "d")
        val index = VariableIndex()
        index.add(a)
        val instance = BitArray(4)
        assertFailsWith<IllegalStateException> {
            a.valueOf(instance, 0, 0)
        }
        instance[1] = true
        assertContentEquals(listOf("b"), a.valueOf(instance, 0, 0)!!)
        instance[0] = true
        assertContentEquals(listOf("a", "b"), a.valueOf(instance, 0, 0)!!)
        instance[2] = true
        assertContentEquals(listOf("a", "b", "c"), a.valueOf(instance, 0, 0)!!)
    }

    @Test
    fun toLiteralOptional() {
        val f = Multiple("a", true, Root(""), "a", "b", "c")
        val index = VariableIndex()
        index.add(f)
        assertEquals(1, f.toLiteral(index))
    }

    @Test
    fun toLiteralMandatory() {
        val f = Multiple("a", false, Root(""), "a", "b", "c")
        val index = VariableIndex()
        index.add(f)
        assertFailsWith(Exception::class) {
            assertEquals(
                Int.MAX_VALUE,
                f.toLiteral(index)
            )
        }
    }

    @Test
    fun toLiteral2() {
        val f = Multiple("", true, Root(""), 1, 2, 3, 4, 5, 6, 7)
        val index = VariableIndex()
        index.add(BitsVar("b", true, Root(""), 5))
        index.add(f)
        assertEquals(7, f.toLiteral(index))
    }

    @Test
    fun missingOptionalTest() {
        val o = Multiple("o", true, Root(""), 1.0, 2.0, 3.0)
        assertFailsWith(IllegalArgumentException::class) {
            o.value(1.2)
        }
    }

    @Test
    fun toLiteralOptionalNot() {
        val a = Nominal("a", true, Root(""), 1.0, 2.0, 3.0)
        val index = VariableIndex()
        index.add(a)
        val o = a.value(1.0).not()
        assertEquals(a, o.canonicalVariable)
        assertEquals(a, a.reifiedValue)
        assertEquals(-2, o.toLiteral(index))
    }

    @Test
    fun toLiteralMandatoryNot() {
        val a = Nominal("a", false, Root(""), 1.0, 2.0, 3.0)
        val index = VariableIndex()
        index.add(a)
        val o = a.value(1.0).not()
        assertEquals(a, o.canonicalVariable)
        assertEquals(-1, o.toLiteral(index))
    }
}
