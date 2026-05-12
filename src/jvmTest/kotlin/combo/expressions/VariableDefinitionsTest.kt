package combo.expressions

import combo.model.VariableIndex
import combo.sat.BitArray
import kotlin.test.*

class FlagTest {

    @Test
    fun index() {
        val f = Flag("", true, Root(""))
        val index = VariableIndex()
        index.add(f)
        assertEquals(0, index.valueIndexOf(f))
        assertEquals(1, f.nbrValues)
    }

    @Test
    fun valueOf() {
        val f = Flag("", true, Root(""))
        val instance = BitArray(1)
        assertNull(f.valueOf(instance, 0, 0))
        instance[0] = true
        assertEquals(true, f.valueOf(instance, 0, 0))
    }

    @Test
    fun toLiteral() {
        val f = Flag("f", 1, Root(""))
        val index = VariableIndex()
        index.add(f)
        assertEquals(1, f.toLiteral(index))
    }

    @Test
    fun toLiteral2() {
        val f = Flag("f", 1, Root(""))
        val index = VariableIndex()
        index.add(BitsVar("b", true, Root(""), 5))
        index.add(f)
        assertEquals(7, f.toLiteral(index))
    }
}
