package combo.model

import combo.expressions.Flag
import combo.expressions.Root
import combo.test.assertContentEquals
import kotlin.test.*


class ScopeTest {

    @Test
    fun find() {
        val scope1 = RootScope(Root("a"))
        val scope2 = scope1.addScope("b", Flag("b", true, scope1.reifiedValue))

        val a = Flag("a", true, scope1.reifiedValue)
        scope1.add(a)
        val b = Flag("b", true, scope2.reifiedValue)
        scope2.add(b)

        assertEquals(a, scope1.find("a")!!)
        assertEquals(b, scope1.find("b")!!)

        assertNull(scope2.find<Flag<*>>("a"))
        assertEquals(b, scope2.find<Flag<*>>("b"))
    }

    @Test
    fun resolve() {
        val scope1 = RootScope(Root("a"))
        val scope2 = scope1.addScope("b", Flag("b", true, scope1.reifiedValue))

        val a = Flag("a", true, scope1.reifiedValue)
        scope1.add(a)
        val b = Flag("b", true, scope2.reifiedValue)
        scope2.add(b)

        assertTrue(scope1.inScope("a"))
        assertEquals(a, scope1.resolve("a"))
        assertFalse(scope1.inScope("b"))
        assertFailsWith(NoSuchElementException::class) {
            scope1.resolve("b")
        }

        assertEquals(a, scope2.resolve("a"))
        assertEquals(b, scope2.resolve("b"))
    }

    @Test
    fun resolveMissing() {
        val scope1 = RootScope(Root("a"))
        val scope2 = scope1.addScope("b", Flag("b", true, scope1.reifiedValue))

        val a = Flag("a", true, scope1.reifiedValue)
        scope1.add(a)
        val b = Flag("b", true, scope2.reifiedValue)
        scope2.add(b)

        assertFalse(scope1.inScope("c"))
        assertFalse(scope2.inScope("c"))
        assertFailsWith(NoSuchElementException::class) {
            scope1.resolve("c")
        }
        assertFailsWith(NoSuchElementException::class) {
            scope2.resolve("c")
        }
    }

    @Test
    fun resolveOverrides() {
        val scope1 = RootScope(Root("a"))
        val scope2 = scope1.addScope("a", Flag("a", true, scope1.reifiedValue))

        val a1 = Flag("a", true, scope1.reifiedValue)
        scope1.add(a1)
        val a2 = Flag("a", true, scope2.reifiedValue)
        scope2.add(a2)

        assertEquals(a1, scope1.resolve("a"))
        assertEquals(a2, scope2.resolve("a"))
    }

    @Test
    fun variableSequence() {
        val scope = RootScope(Root(""))
        var prev: Scope = scope
        val indices = Array(10) {
            val f = Flag("$it", true, scope.reifiedValue)
            prev.add(f)
            prev = prev.addScope("$it", f)
            prev
        }

        val rootExpected = listOf(*((0 until 10).toList().toTypedArray()))
        assertEquals(rootExpected, scope.asSequence().toList().map { it.name.toInt() })
        for (i in 0 until 10) {
            val expected = listOf(*(((i + 1) until 10).toList().toTypedArray()))
            assertEquals(expected, indices[i].asSequence().toList().map { it.name.toInt() })
        }
    }

    @Test
    fun variableSequence2() {

        val scope0 = RootScope(Root(""))
        val f0 = Flag("0", true, scope0.reifiedValue)
        scope0.add(f0)

        val f1 = Flag("1", true, scope0.reifiedValue)
        scope0.add(f1)
        val scope1 = scope0.addScope("", f1)

        val f2 = Flag("2", true, scope1.reifiedValue)
        scope1.add(f2)
        val scope2 = scope1.addScope("", f2)

        val f3 = Flag("3", true, scope2.reifiedValue)
        scope2.add(f3)
        val scope3 = scope2.addScope("", f3)

        val f4 = Flag("4", true, scope3.reifiedValue)
        scope3.add(f4)

        val scope4 = scope1.addScope("", f4)
        scope4.add(Flag("5", true, scope4.reifiedValue))
        scope4.add(Flag("6", true, scope4.reifiedValue))

        val f5 = Flag("5", true, scope2.reifiedValue)
        val scope5 = scope2.addScope("", f5)
        scope5.add(Flag("7", true, scope5.reifiedValue))
        scope5.add(Flag("8", true, scope5.reifiedValue))
        scope5.add(Flag("9", true, scope5.reifiedValue))

        assertContentEquals(intArrayOf(0, 1, 2, 5, 6, 3, 7, 8, 9, 4), scope0.asSequence().toList().map { it.name.toInt() }.toIntArray())
        assertContentEquals(intArrayOf(2, 5, 6, 3, 7, 8, 9, 4), scope1.asSequence().toList().map { it.name.toInt() }.toIntArray())
        assertContentEquals(intArrayOf(3, 7, 8, 9, 4), scope2.asSequence().toList().map { it.name.toInt() }.toIntArray())
        assertContentEquals(intArrayOf(4), scope3.asSequence().toList().map { it.name.toInt() }.toIntArray())
        assertContentEquals(intArrayOf(5, 6), scope4.asSequence().toList().map { it.name.toInt() }.toIntArray())
        assertContentEquals(intArrayOf(7, 8, 9), scope5.asSequence().toList().map { it.name.toInt() }.toIntArray())
    }
}
