package combo.model

import combo.expressions.Flag
import combo.expressions.Nominal
import combo.expressions.Root
import kotlin.test.Test
import kotlin.test.assertEquals

class VariableIndexTest {

    @Test
    fun add() {
        val index = VariableIndex()
        val v1 = Flag("a", true, Root(""))
        index.add(v1)
        assertEquals(1, index.nbrValues)
        assertEquals(0, index.valueIndexOf(v1))
        val v2 = Nominal("a", true, Root(""), 1, 2, 3)
        index.add(v2)
        assertEquals(5, index.nbrValues)
        assertEquals(1, index.valueIndexOf(v2))
    }

}
