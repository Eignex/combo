package combo.model

import combo.math.permutation
import combo.sat.Problem
import combo.expressions.Conjunction
import combo.expressions.Multiple
import combo.expressions.Relation
import combo.util.collectionOf
import kotlin.random.Random

object TestModels {

    val MODELS: List<Model> by lazy { listOf(MODEL1, MODEL2, MODEL3, MODEL4, MODEL5, MODEL6, MODEL7) }
    val LARGE_MODELS: List<Model> by lazy { listOf(LARGE1, LARGE2, LARGE3) }
    val UNSAT_MODELS: List<Model> by lazy { listOf(UNSAT1, UNSAT2, UNSAT3, UNSAT4) }
    val NUMERIC_MODELS: List<Model> by lazy { listOf(NUMERIC1, NUMERIC2, NUMERIC3) }
    val CSP_MODELS: List<Model> by lazy { listOf(CSP1) }
    val TINY_MODELS: List<Model> by lazy { listOf(MODEL2, MODEL3) }

    val SAT_PROBLEMS: List<Problem> by lazy { MODELS.map { m -> m.problem } }
    val UNSAT_PROBLEMS: List<Problem> by lazy { UNSAT_MODELS.map { m -> m.problem } }
    val LARGE_PROBLEMS: List<Problem> by lazy { LARGE_MODELS.map { m -> m.problem } }
    val NUMERIC_PROBLEMS: List<Problem> by lazy { NUMERIC_MODELS.map { m -> m.problem } }
    val CSP_PROBLEMS: List<Problem> by lazy { CSP_MODELS.map { m -> m.problem } }
    val TINY_PROBLEMS: List<Problem> by lazy { TINY_MODELS.map { m -> m.problem } }

    val MODEL1 by lazy {
        Model.model("All Round Basic") {
            bool("f1")
            model("f2") {
                val m1 = multiple("m1", 4, 5, 6)
                impose {
                    m1.value(4) implies m1.value(6)
                }
                model(optionalNominal("alt1", "a", "c")) {
                    val f3 = bool("f3")
                    val alt2 = optionalNominal("alt2", "a", "d")
                    impose {
                        f3 implies alt2.value("d")
                    }
                }
            }
            flag("f4", true)
        }
    }

    val MODEL2 by lazy {
        Model.model("All Unit Features") {
            val f2 = bool("f2")
            impose { f2 }
            model("f1") {
                optionalNominal("alt1", "a", "c")
                optionalMultiple("or1", 4, 5, 6)
                impose { !(f2 and "or1") }
                impose { !(f2 and "alt1") }
            }
            impose { "f2" implies "f1" }
            bool("f3")
            impose { !(f2 and "f3") }
        }
    }

    val MODEL3 by lazy {
        Model.model("Unit Options in Select") {
            val alt1 = nominal("alt1", "a", "b", "c")
            val mult1 = multiple("mult1", "a", "b", "c")
            impose { mult1.value("b") }
            impose { alt1.value("c") }
        }
    }

    val MODEL4 by lazy {
        Model.model("All Standard Features") {
            bool()
            optionalNominal("a1", 1, 2, 3)
            optionalMultiple("m1", values = arrayOf(1, 2, 3))
            nominal("a2", values = arrayOf(1, 2, 3))
            multiple("m2", values = arrayOf(1, 2, 3))
        }
    }

    val MODEL5 by lazy {
        Model.model("Multiple Sub Models") {
            bool("b1")
            bool("b2")
            bool("b3")
            model("r1") {
                model("sub1") {
                    model("sub2") {
                        model("sub3") {
                            bool("sub4")
                        }
                    }
                }
            }
            model("r2") {
                bool("f1")
            }
            impose { this@model["f1"] or this@model["sub4"] }
        }
    }

    val MODEL6 by lazy {
        Model.model("All Cardinality Options except NE") {
            val flags = Array(12) { bool("$it") }
            impose { atMost(4, *flags.sliceArray(0 until 10)) }
            impose { atLeast(4, *flags) }
            impose { exactly(2, flags[0], flags[1], flags[2]) }
            impose { excludes(flags[2], flags[3]) }
            impose { cardinality(2, Relation.GT, *flags) }
            impose { cardinality(8, Relation.LT, *flags) }
        }
    }

    val MODEL7 by lazy {
        Model.model("Diverging sub-models") {
            val top = this
            model("sub1") {
                bool("f1")
            }
            model("sub2") {
                model(top["f1"], "r") {
                    flag("f2", 1)
                }
            }
        }
    }

    val UNSAT1 by lazy {
        Model.model("Simple 2-Unsat") {
            val f1 = bool()
            val f2 = bool()
            impose { f1 or f2 }
            impose { f1 or !f2 }
            impose { !f1 or f2 }
            impose { !f1 or !f2 }
        }
    }

    val UNSAT2 by lazy {
        Model.model("Simple 3-Unsat") {
            val f1 = bool()
            val f2 = bool()
            val f3 = bool()
            impose { f1 or f2 or f3 }
            impose { f1 or f2 or !f3 }
            impose { f1 or !f2 or f3 }
            impose { f1 or !f2 or !f3 }
            impose { !f1 or f2 or f3 }
            impose { !f1 or f2 or !f3 }
            impose { !f1 or !f2 or f3 }
            impose { !f1 or !f2 or !f3 }
        }
    }

    val UNSAT3 by lazy {
        Model.model("Unsat by Cardinality") {
            val a = optionalNominal(values = (1..10).toList().toTypedArray())
            impose { atLeast(2, a.value(1), a.value(2), a.value(3)) }
        }
    }

    val UNSAT4 by lazy {
        Model.model("Modified Simple 2-Unsat") {
            val a = bool()
            val b = bool()
            impose { a xor b }
            model(b) {
                val c = bool()
                impose { a equivalent c }
            }
            impose { !b or a }
        }
    }
    val LARGE1 by lazy {
        Model.model("Just Alternatives") {
            optionalNominal(values = Array(80) { it })
            multiple(values = Array(80) { it })
            nominal(values = Array(80) { it })
            optionalMultiple(values = Array(80) { it })
        }
    }

    val LARGE2 by lazy {
        Model.model("Hierarchical Category Model") {
            val topCategories = Array(5) { i ->
                model("Cat $i") {
                    val leafCategories = Array(10) { j ->
                        optionalMultiple("Cat $i $j", *Array(10) { 1 + it })
                    }
                    impose { this.scope.reifiedValue reifiedEquivalent disjunction(*leafCategories) }
                }
            }
            impose { disjunction(*topCategories) }
            impose {
                atMost(5, *scope.asSequence()
                        .filter { it is Multiple<*> }
                        .flatMap { (it as Multiple<*>).values.asSequence() }
                        .toList().toTypedArray())
            }
        }
    }

    val LARGE3 by lazy {
        Model.model("Flag chain") {
            var next: Model.ModelBuilder<*> = this
            for (k in 1..500)
                next = next.model("$k") {}
        }
    }

    val LARGE4 by lazy {
        Model.model("Random Disjunctions") {
            val flags = Array(500) { bool("$it") }
            val rng = Random(0)
            for (i in 1..1000) {
                impose {
                    disjunction(*permutation(flags.size, rng).asSequence()
                            .take(rng.nextInt(8) + 2)
                            .map { flags[it] }
                            .map { if (rng.nextBoolean()) it.not() else it }
                            .toList().toTypedArray())
                }
            }
        }
    }

    val NUMERIC1 by lazy {
        Model.model("Ints in hierarchy") {
            optionalInt("int1", -1, 1)
            int("int2", 2, 3)
            optionalInt("int3", 0, Int.MAX_VALUE)
            model {
                optionalInt("int4")
                bool()
                int("int5")
            }
        }
    }

    val NUMERIC2 by lazy {
        Model.model("Ints with constraints") {
            optionalInt()
            optionalInt()
            optionalInt()
            addConstraint(Conjunction(collectionOf(4, -42, 99)))
        }
    }

    val NUMERIC3 by lazy {
        Model.model("All kinds of numeric variables") {
            optionalInt("int1", min = -100, max = 100)
            int("int2", min = -10, max = 0)
            optionalFloat("float1", min = -0.1f, max = 1.0f)
            float("float2", min = 0.0f, max = 1.0f)
            optionalBits(nbrBits = 10)
            bits(nbrBits = 100)
            model {
                optionalInt("int3", min = -10, max = 100)
                int("int4")
                optionalFloat("float3", min = -0.1f, max = 1.0f)
                float("float4", min = -0.1f, max = 1.0f)
                optionalBits(nbrBits = 10)
                bits(nbrBits = 100)
            }
        }
    }

    val CSP1 by lazy {
        Model.model("All kinds of PB constraints") {
            val f1 = bool()
            val f2 = bool()
            val f3 = bool()
            val f4 = bool()
            impose {
                f1 reifiedImplies conjunction(f2, f3)
            }
            impose {
                f4 reifiedImplies disjunction(f1, f2, f3)
            }
            impose {
                cardinality(3, Relation.LT, !f1, f2, f3, !f4)
            }
            impose {
                linear(2, Relation.GT, intArrayOf(1, 2, 3, 4), arrayOf(f1, f2, f3, f4))
            }
        }
    }
}
