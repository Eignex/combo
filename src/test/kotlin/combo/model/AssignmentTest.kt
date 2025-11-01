package combo.model

import combo.expressions.BitsVar
import combo.expressions.Flag
import combo.expressions.FloatVar
import combo.expressions.IntVar
import combo.expressions.Multiple
import combo.expressions.Nominal
import combo.sat.BitArray
import combo.sat.optimizers.ExhaustiveSolver
import combo.test.assertContentEquals
import combo.util.MIN_VALUE32
import kotlin.test.*

class AssignmentTest {

    @Test
    fun getMissingVariable() {
        val m = Model.model {
            flag("a", 10)
        }
        m.toAssignment(BitArray(1, IntArray(1))).get<Int>("b")
        assertFailsWith(NoSuchElementException::class) {
            m.toAssignment(BitArray(1, IntArray(1))).getOrThrow<Int>("b")
        }
    }

    @Test
    fun getOrThrowThrowsOnNull() {
        assertFailsWith(NoSuchElementException::class) {
            val m = Model.model {
                flag("f", 10)
            }
            m.toAssignment(BitArray(1, IntArray(1))).getOrThrow<Int>("f")
        }
        assertFailsWith(NoSuchElementException::class) {
            val m = Model.model {
                flag("f", 10)
            }
            val f: Flag<Int> = m.scope.find("f")!!
            m.toAssignment(BitArray(1, IntArray(1))).getOrThrow(f)
        }
    }

    @Test
    fun getFlag() {
        val m = Model.model {
            flag("f", 10)
        }
        val f: Flag<Int> = m.scope.find("f")!!
        assertEquals(10, m.toAssignment(BitArray(1, IntArray(1) { 0b1 }))[f])
        assertEquals(null, m.toAssignment(BitArray(1, IntArray(1) { 0b0 }))[f])
    }

    @Test
    fun getOr() {
        val m = Model.model {
            multiple("m", 10, 20)
        }
        val f: Multiple<Int> = m.scope.find("m")!!
        assertContentEquals(listOf(10), m.toAssignment(BitArray(2, IntArray(1) { 0b01 }))[f]!!)
        assertContentEquals(listOf(10, 20), m.toAssignment(BitArray(2, IntArray(1) { 0b11 }))[f]!!)
    }

    @Test
    fun getAlternative() {
        val m = Model.model {
            optionalNominal("a", "s1", "s2")
        }
        val f: Nominal<String> = m.scope.find("a")!!
        assertNull(m.toAssignment(BitArray(3, IntArray(1) { 0b000 }))[f])
        assertEquals("s1", m.toAssignment(BitArray(3, IntArray(3) { 0b011 }))[f])
        assertEquals("s2", m.toAssignment(BitArray(3, IntArray(3) { 0b101 }))[f])
    }

    @Test
    fun getIntVar() {
        val m = Model.model {
            optionalInt("i", 10, 20)
        }
        val i: IntVar = m.scope.find("i")!!
        assertEquals(12, m.toAssignment(BitArray(6, IntArray(1) { 0b011001 }))[i])
        assertEquals(20, m.toAssignment(BitArray(6, IntArray(1) { 0b101001 }))[i])
        assertEquals(null, m.toAssignment(BitArray(6, IntArray(1) { 0b000000 }))[i])
    }

    @Test
    fun getFloatVar() {
        val m = Model.model {
            float("f")
        }
        val f: FloatVar = m.scope.find("f")!!
        assertEquals(MIN_VALUE32.toBits(), m.toAssignment(BitArray(32, IntArray(1) { 0b1 }))[f]!!.toBits())
        assertEquals(0.0f, m.toAssignment(BitArray(32, IntArray(1) { 0b0 }))[f])
    }

    @Test
    fun getBitsVar() {
        val m = Model.model {
            optionalBits("b", 5)
        }
        val b: BitsVar = m.scope.find("b")!!
        for (i in 0 until 32) {
            val toAssignment = m.toAssignment(BitArray(6, IntArray(1) { (i shl 1) or 1 }))
            assertEquals(BitArray(5, IntArray(1) { i }), toAssignment[b])
        }
        assertEquals(null, m.toAssignment(BitArray(6, IntArray(1) { 0b0 }))[b])
    }

    @Test
    fun getString() {
        val m = Model.model {
            flag("a", "b")
        }
        assertEquals("b", m.toAssignment(BitArray(1, IntArray(1) { 0b1 })).getString("a"))
        assertEquals("", m.toAssignment(BitArray(1, IntArray(1) { 0b0 })).getString("a"))

        val a: Flag<String> = m.scope.find("a")!!
        assertEquals("b", m.toAssignment(BitArray(1, IntArray(1) { 0b1 })).getString(a))
        assertEquals("", m.toAssignment(BitArray(1, IntArray(1) { 0b0 })).getString(a))
    }

    @Test
    fun getChar() {
        val m = Model.model {
            flag("a", 'b')
        }
        assertEquals('b', m.toAssignment(BitArray(1, IntArray(1) { 0b1 })).getChar("a"))
        assertEquals('\u0000', m.toAssignment(BitArray(1, IntArray(1) { 0b0 })).getChar("a"))

        val a: Flag<Char> = m.scope.find("a")!!
        assertEquals('b', m.toAssignment(BitArray(1, IntArray(1) { 0b1 })).getChar(a))
        assertEquals('\u0000', m.toAssignment(BitArray(1, IntArray(1) { 0b0 })).getChar(a))
    }

    @Test
    fun getBool() {
        val m = Model.model {
            bool("a")
        }
        assertEquals(true, m.toAssignment(BitArray(1, IntArray(1) { 0b1 })).getBoolean("a"))
        assertEquals(false, m.toAssignment(BitArray(1, IntArray(1) { 0b0 })).getBoolean("a"))

        val b: Flag<Boolean> = m.scope.find("a")!!
        assertEquals(true, m.toAssignment(BitArray(1, IntArray(1) { 0b1 })).getBoolean(b))
        assertEquals(false, m.toAssignment(BitArray(1, IntArray(1) { 0b0 })).getBoolean(b))
    }

    @Test
    fun getLong() {
        val m = Model.model {
            flag("l", 1L)
        }
        assertEquals(1L, m.toAssignment(BitArray(1, IntArray(1) { 0b1 })).getLong("l"))
        assertEquals(0L, m.toAssignment(BitArray(1, IntArray(1) { 0b0 })).getLong("l"))

        val l: Flag<Long> = m.scope.find("l")!!
        assertEquals(1L, m.toAssignment(BitArray(1, IntArray(1) { 0b1 })).getLong(l))
        assertEquals(0L, m.toAssignment(BitArray(1, IntArray(1) { 0b0 })).getLong(l))
    }

    @Test
    fun getInt() {
        val m = Model.model {
            optionalNominal("i", 1, 2, 3, 4)
        }
        assertEquals(3, m.toAssignment(BitArray(5, IntArray(1) { 0b01001 })).getInt("i"))
        assertEquals(0, m.toAssignment(BitArray(5, IntArray(1) { 0b00000 })).getInt("i"))

        val i: Nominal<Int> = m.scope.find("i")!!
        assertEquals(1, m.toAssignment(BitArray(5, IntArray(1) { 0b00011 })).getInt(i))
        assertEquals(0, m.toAssignment(BitArray(5, IntArray(1) { 0b00000 })).getInt(i))
    }

    @Test
    fun getShort() {
        val s1: Short = 0
        val s2: Short = 2
        val m = Model.model {
            flag("s", s2)
        }
        assertEquals(s2, m.toAssignment(BitArray(1, IntArray(1) { 0b1 })).getShort("s"))
        assertEquals(s1, m.toAssignment(BitArray(1, IntArray(1) { 0b0 })).getShort("s"))

        val s: Flag<Short> = m.scope.find("s")!!
        assertEquals(s2, m.toAssignment(BitArray(1, IntArray(1) { 0b1 })).getShort(s))
        assertEquals(s1, m.toAssignment(BitArray(1, IntArray(1) { 0b0 })).getShort(s))
    }

    @Test
    fun getByte() {
        val b1: Byte = 0
        val b2: Byte = 2
        val m = Model.model {
            flag("b", b2)
        }
        assertEquals(b2, m.toAssignment(BitArray(1, IntArray(1) { 0b1 })).getByte("b"))
        assertEquals(b1, m.toAssignment(BitArray(1, IntArray(1) { 0b0 })).getByte("b"))

        val b: Flag<Byte> = m.scope.find("b")!!
        assertEquals(b2, m.toAssignment(BitArray(1, IntArray(1) { 0b1 })).getByte(b))
        assertEquals(b1, m.toAssignment(BitArray(1, IntArray(1) { 0b0 })).getByte(b))
    }

    @Test
    fun getDouble() {
        val m = Model.model {
            flag("d", 0.1)
        }
        assertEquals(0.1, m.toAssignment(BitArray(1, IntArray(1) { 0b1 })).getDouble("d"))
        assertEquals(0.0, m.toAssignment(BitArray(1, IntArray(1) { 0b0 })).getDouble("d"))

        val d: Flag<Double> = m.scope.find("d")!!
        assertEquals(0.1, m.toAssignment(BitArray(1, IntArray(1) { 0b1 })).getDouble(d))
        assertEquals(0.0, m.toAssignment(BitArray(1, IntArray(1) { 0b0 })).getDouble(d))
    }

    @Test
    fun getFloat() {
        val m = Model.model {
            flag("f", 0.1f)
        }
        assertEquals(0.1f, m.toAssignment(BitArray(1, IntArray(1) { 0b1 })).getFloat("f"))
        assertEquals(0.0f, m.toAssignment(BitArray(1, IntArray(1) { 0b0 })).getFloat("f"))

        val f: Flag<Float> = m.scope.find("f")!!
        assertEquals(0.1f, m.toAssignment(BitArray(1, IntArray(1) { 0b1 })).getFloat(f))
        assertEquals(0.0f, m.toAssignment(BitArray(1, IntArray(1) { 0b0 })).getFloat(f))
    }

    @Test
    fun asSequence() {
        val m = TestModels.MODEL1
        val solver = ModelOptimizer(m, ExhaustiveSolver(m.problem))
        val a = solver.witnessOrThrow(m.scope.find<Multiple<Int>>("m1")!!.value(4))
        val iterated = a.asSequence().toList()
        assertTrue(iterated.size >= 2)
        assertTrue(iterated.contains(Assignment.VariableAssignment(m["f2"], true)))
        assertNotNull(iterated.find { it.variable == m["m1"] })
    }
}
