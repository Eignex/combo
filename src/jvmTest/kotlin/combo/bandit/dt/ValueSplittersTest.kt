package combo.bandit.dt

import combo.model.Model.Companion.model
import combo.model.TestModels
import combo.sat.toIx
import combo.util.IntHashSet
import combo.util.intListOf
import kotlin.random.Random
import kotlin.test.Ignore
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue


class ValueSplittersTest {

    @Test
    fun flagSplitterNoReified() {
        val model = model { bool("f") }
        val splitter = defaultValueSplitter(model, model["f"])
        assertEquals(0, splitter.nextSplit(intListOf(), intListOf(), Random))
        assertEquals(0, splitter.nextSplit(intListOf(13), intListOf(), Random))
        assertEquals(0, splitter.nextSplit(intListOf(), intListOf(1), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(), intListOf(0), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(-1), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(1), intListOf(), Random))
    }

    @Test
    fun flagSplitterReified() {
        val model = model { model { bool("f") } }
        val splitter = defaultValueSplitter(model, model["f"])
        assertEquals(0, splitter.nextSplit(intListOf(), intListOf(), Random))
        assertEquals(0, splitter.nextSplit(intListOf(), intListOf(2), Random))
        assertEquals(1, splitter.nextSplit(intListOf(), intListOf(0), Random))
        assertEquals(1, splitter.nextSplit(intListOf(1), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(-1), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(-2), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(), intListOf(0, 1), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(1, -2), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(1, 2), intListOf(), Random))
    }

    @Test
    fun standardSplitterNoReifiedMandatory() {
        val model = model { nominal("n", 1, 2, 3) }
        val splitter = defaultValueSplitter(model, model["n"])
        assertTrue(splitter.nextSplit(intListOf(), intListOf(), Random) in 0..2)
        assertTrue(splitter.nextSplit(intListOf(), intListOf(1), Random) in 0..2)
        assertEquals(2, splitter.nextSplit(intListOf(), intListOf(0, 1), Random))
        assertEquals(2, splitter.nextSplit(intListOf(-2), intListOf(0), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(), intListOf(0, 1, 2), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(-1, -2, -3), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(1, 2, 3), intListOf(), Random))
    }

    @Test
    fun standardSplitterReifiedMandatory() {
        val model = model { model { nominal("n", 1, 2, 3) } }
        val splitter = defaultValueSplitter(model, model["n"])
        assertEquals(0, splitter.nextSplit(intListOf(), intListOf(), Random))
        assertEquals(0, splitter.nextSplit(intListOf(-2), intListOf(), Random))
        assertTrue(splitter.nextSplit(intListOf(), intListOf(0), Random) in 1..3)
        assertTrue(splitter.nextSplit(intListOf(1), intListOf(), Random) in 1..3)
        assertEquals(3, splitter.nextSplit(intListOf(), intListOf(0, 1, 2), Random))
        assertEquals(2, splitter.nextSplit(intListOf(1, -2, -4), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(1, -2, -4), intListOf(2), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(-1), intListOf(), Random))
    }

    @Test
    fun standardSplitterNoReifiedOptional() {
        val model = model { optionalNominal("n", 1, 2, 3) }
        val splitter = defaultValueSplitter(model, model["n"])
        assertEquals(0, splitter.nextSplit(intListOf(), intListOf(), Random))
        assertEquals(1, splitter.nextSplit(intListOf(1), intListOf(2, 3), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(-1), intListOf(), Random))
    }

    @Test
    fun standardSplitterReifiedOptional() {
        val model = model { model { optionalMultiple("m", 1, 2, 3) } }
        val splitter = defaultValueSplitter(model, model["m"])
        assertEquals(0, splitter.nextSplit(intListOf(), intListOf(), Random))
        assertEquals(1, splitter.nextSplit(intListOf(1), intListOf(), Random))
        assertTrue(splitter.nextSplit(intListOf(1), intListOf(1), Random) in 2..4)
        assertEquals(-1, splitter.nextSplit(intListOf(-1), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(1, -2), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(-2), intListOf(), Random))
    }

    @Test
    fun floatSplitterContradiction() {
        val model = model { float("f", 10.0f, 10.9f) }
        val splitter = defaultValueSplitter(model, model["f"])
        assertEquals(-1, splitter.nextSplit(intListOf(21), intListOf(), Random))
        assertEquals(19, splitter.nextSplit(intListOf(-21), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(-22), intListOf(), Random))
        assertEquals(19, splitter.nextSplit(intListOf(22), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(24), intListOf(), Random))
        assertEquals(19, splitter.nextSplit(intListOf(-24), intListOf(), Random))
        assertEquals(19, splitter.nextSplit(intListOf(-30), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(30), intListOf(), Random))
    }

    @Test
    fun floatSplitterExhaustiveAudit() {
        val model = model { model { optionalFloat("f", -1000f, 201240f) } }
        val splitter = defaultValueSplitter(model, model["f"])
        val set = IntHashSet(nullValue = -1)
        for (i in 32 downTo 1) {
            assertEquals(1 + i, splitter.nextSplit(intListOf(1, 2), set, Random))
            set.add(1 + i)
        }
        assertEquals(-1, splitter.nextSplit(intListOf(1, 2), set, Random))
    }

    @Test
    fun floatSplitterNoReifiedMandatory1() {
        val model = model { float("f", 0f, 1f) }
        val splitter = defaultValueSplitter(model, model["f"])
        assertEquals(29, splitter.nextSplit(intListOf(), intListOf(), Random))
    }

    @Test
    fun floatSplitterNoReifiedMandatory2() {
        val model = model { float("f", -1f, 1f) }
        val splitter = defaultValueSplitter(model, model["f"])
        assertEquals(31, splitter.nextSplit(intListOf(), intListOf(), Random))
    }

    @Test
    fun floatSplitterReifiedMandatory() {
        val model = model { model { float("f") } }
        val splitter = defaultValueSplitter(model, model["f"])
        assertEquals(0, splitter.nextSplit(intListOf(), intListOf(), Random))
        assertEquals(32, splitter.nextSplit(intListOf(1), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(-1), intListOf(), Random))
    }

    @Test
    fun floatSplitterNoReifiedOptional() {
        val model = model { optionalFloat("f", -1.4f, 20.1f) }
        val splitter = defaultValueSplitter(model, model["f"])
        assertEquals(0, splitter.nextSplit(intListOf(), intListOf(), Random))
        assertEquals(32, splitter.nextSplit(intListOf(), intListOf(0), Random))
        assertEquals(32, splitter.nextSplit(intListOf(1), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(-1), intListOf(), Random))
    }

    @Test
    fun floatSplitterReifiedOptional() {
        val model = model { model { optionalFloat("f", -1.4f, 20.1f) } }
        val splitter = defaultValueSplitter(model, model["f"])
        assertEquals(0, splitter.nextSplit(intListOf(), intListOf(), Random))
        assertEquals(1, splitter.nextSplit(intListOf(1), intListOf(), Random))
        assertEquals(33, splitter.nextSplit(intListOf(1), intListOf(1), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(-1), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(1, -2), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(-2), intListOf(), Random))
    }

    @Test
    @Ignore
    fun intSplitterContradiction() {
        val model = model { int("i", 2, 3) }
        val splitter = defaultValueSplitter(model, model["i"])
        assertEquals(1, splitter.nextSplit(intListOf(), intListOf(), Random))
        assertEquals(1, splitter.nextSplit(intListOf(-2), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(), intListOf(3, 4), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(4, 5), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(-4, 5), intListOf(), Random))
    }

    @Test
    fun intSplitterNoReifiedMandatory() {
        val model = model { int("i", 2, 3) }
        val splitter = defaultValueSplitter(model, model["i"])
        assertEquals(1, splitter.nextSplit(intListOf(), intListOf(), Random))
        assertEquals(1, splitter.nextSplit(intListOf(), intListOf(0), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(), intListOf(0, 1), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(1, 2), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(-1, 2), intListOf(), Random))
    }

    @Test
    fun intSplitterNoReifiedOptional() {
        val model = model { optionalInt("i", -1, 2) }
        val splitter = defaultValueSplitter(model, model["i"])
        assertEquals(0, splitter.nextSplit(intListOf(), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(-1), intListOf(), Random))
        assertEquals(3, splitter.nextSplit(intListOf(), intListOf(0), Random))
        assertEquals(2, splitter.nextSplit(intListOf(), intListOf(0, 3), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(), intListOf(0, 1, 2, 3), Random))
        assertEquals(3, splitter.nextSplit(intListOf(1), intListOf(), Random))
        assertEquals(2, splitter.nextSplit(intListOf(1, 4), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(1, 2, -3, 4), intListOf(), Random))
        assertEquals(1, splitter.nextSplit(intListOf(1), intListOf(2, 3), Random))
    }

    @Test
    fun intSplitterReifiedMandatory() {
        val model = model { model { int("i") } }
        val splitter = defaultValueSplitter(model, model["i"])
        assertEquals(0, splitter.nextSplit(intListOf(), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(-1), intListOf(), Random))
        assertEquals(32, splitter.nextSplit(intListOf(1), intListOf(), Random))
        assertEquals(32, splitter.nextSplit(intListOf(), intListOf(0), Random))
    }

    @Test
    fun intSplitterReifiedOptional() {
        val model = model { model { optionalInt("i") } }
        val splitter = defaultValueSplitter(model, model["i"])
        assertEquals(0, splitter.nextSplit(intListOf(), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(-1), intListOf(), Random))
        assertEquals(-1, splitter.nextSplit(intListOf(-2), intListOf(), Random))
        assertEquals(33, splitter.nextSplit(intListOf(1), intListOf(1), Random))

    }

    @Test
    fun splittingExhaustive() {
        // For each model, for each variable
        for (model in TestModels.MODELS + TestModels.NUMERIC_MODELS) {
            val total = IntHashSet(nullValue = -1)
            for (variable in model.index) {
                val varIx = model.index.valueIndexOf(variable)
                val range = varIx..varIx + variable.nbrValues
                val splitter = defaultValueSplitter(model, variable)
                val set = IntHashSet(nullValue = 0)
                val values = IntHashSet(nullValue = -1)
                while (true) {
                    val ix = splitter.nextSplit(set, values, Random)
                    if (ix >= 0) {
                        assertTrue(ix in range || ix == variable.parentLiteral(model.index).toIx())
                        values.add(ix)
                        total.add(ix)
                    } else break
                }
            }
        }
    }
}