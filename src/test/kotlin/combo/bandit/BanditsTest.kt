package combo.bandit

import combo.bandit.dt.ForestData
import combo.bandit.dt.TreeData
import combo.bandit.glm.*
import combo.bandit.univariate.*
import combo.math.*
import combo.model.Model
import combo.model.Model.Companion.model
import combo.model.TestModels
import combo.model.TestModels.MODEL1
import combo.model.TestModels.MODELS
import combo.sat.BitArray
import combo.sat.Instance
import combo.sat.ValidationException
import combo.sat.constraints.Conjunction
import combo.util.IntCollection
import combo.util.collectionOf
import combo.util.mapArray
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.*

abstract class BanditTest<B : Bandit<*>> {
    abstract fun bandit(model: Model, parameters: TestParameters): B
    open fun infeasibleBandit(model: Model, parameters: TestParameters): B? =
            bandit(model, parameters)

    @Test
    fun smallInfeasibleProblem() {
        for ((i, m) in TestModels.UNSAT_MODELS.withIndex()) {
            try {
                val infeasibleBandit = infeasibleBandit(m, TestParameters(
                        type = TestType.random(), maximize = Random.nextBoolean()))
                if (infeasibleBandit != null) {
                    assertFailsWith(ValidationException::class, "Model $i") {
                        infeasibleBandit.chooseOrThrow()
                    }
                }
            } catch (e: ValidationException) {
            }
        }
    }

    @Test
    fun poissonDefault() {
        val m = model { bool() }
        val bandit = bandit(m, TestParameters(TestType.POISSON, 0, true))
        bandit.update(BitArray(1), 0f)
    }

    @Test
    fun allProblemsFeasible() {
        for ((i, m) in MODELS.withIndex()) {
            val seed = System.currentTimeMillis().toInt()
            val parameters = TestParameters(randomSeed = seed)
            val b = bandit(m, parameters)
            val rng = Random(seed)
            for (j in 0 until 10) {
                val instance = b.chooseOrThrow()
                assertTrue(m.problem.satisfies(instance), "Model $i")
                b.update(instance, parameters.type.linearRewards(instance, rng))
            }
        }
    }

    @Test
    fun minimizeVsMaximize() {
        val m = MODEL1
        val bandit1 = bandit(m, TestParameters(TestType.BINOMIAL, 1, true))
        val bandit2 = bandit(m, TestParameters(TestType.BINOMIAL, 2, false))
        val rng = Random(1)

        for (i in 1..50) {
            val instance1 = bandit1.chooseOrThrow()
            val instance2 = bandit2.chooseOrThrow()
            assertTrue(m.problem.satisfies(instance1))
            assertTrue(m.problem.satisfies(instance2))
            val r1 = TestType.BINOMIAL.linearRewards(instance1, rng)
            bandit1.update(instance1, r1)
            val r2 = TestType.BINOMIAL.linearRewards(instance2, rng)
            bandit2.update(instance2, r2)
        }
        val sum1 = bandit1.rewards.values().sum()
        val sum2 = bandit2.rewards.values().sum()

        assertTrue(sum1 + 10 > sum2 - 10)
    }

    @Test
    fun minimizeVsMaximizeBatch() {
        val m = MODEL1
        val bandit1 = bandit(m, TestParameters(TestType.BINOMIAL, 1, true))
        val bandit2 = bandit(m, TestParameters(TestType.BINOMIAL, 2, false))
        val rng = Random(1)

        for (i in 1..10) {
            val instances1 = Array(10) { bandit1.chooseOrThrow() }
            val instances2 = Array(10) { bandit2.chooseOrThrow() }
            val results1 = instances1.mapArray { TestType.BINOMIAL.linearRewards(it, rng) }.toFloatArray()
            val results2 = instances1.mapArray { TestType.BINOMIAL.linearRewards(it, rng) }.toFloatArray()
            val weights1 = FloatArray(10) { (rng.nextInt(5) + 1).toFloat() }
            val weights2 = FloatArray(10) { (rng.nextInt(5) + 1).toFloat() }
            bandit1.updateAll(instances1, results1, weights1)
            bandit1.updateAll(instances2, results2, weights2)
        }
        val sum1 = bandit1.rewards.values().sum()
        val sum2 = bandit2.rewards.values().sum()
        assertTrue(sum1 + 10 > sum2 - 10)
    }

    @Test
    fun assumptionsSatisfied() {
        val m = TestModels.MODEL4
        val bandit = bandit(m, TestParameters(TestType.POISSON, rewards = BucketSample(4)))
        for (i in 1..10) {
            val instance = if (Random.nextBoolean()) bandit.chooseOrThrow(collectionOf(3, 12)).also {
                assertTrue { Conjunction(collectionOf(3, 12)).satisfies(it) }
            }
            else bandit.chooseOrThrow()
            assertTrue(m.problem.satisfies(instance))
            bandit.update(instance, TestType.POISSON.linearRewards(instance, Random))
        }
    }

    @Test
    fun assumptionsToInfeasible() {
        fun testInfeasible(assumptions: IntCollection, m: Model) {
            assertFailsWith(ValidationException::class) {
                val bandit = bandit(m, TestParameters(TestType.NORMAL))
                bandit.chooseOrThrow(assumptions)
            }
        }
        testInfeasible(collectionOf(11, 12), MODEL1)
        testInfeasible(collectionOf(6, -7, -8), MODEL1)
        testInfeasible(collectionOf(-2, 4), MODEL1)
        testInfeasible(collectionOf(1, 6), TestModels.MODEL3)
        testInfeasible(collectionOf(3, 4, 5), TestModels.MODEL4)
        testInfeasible(collectionOf(-10, -11, -12), TestModels.MODEL4)
        testInfeasible(collectionOf(-4, 5), TestModels.MODEL5)
    }

    @Test
    fun randomSeedDeterministic() {
        val bandit1 = bandit(TestModels.MODEL3, TestParameters(TestType.NORMAL, randomSeed = 0))
        val bandit2 = bandit(TestModels.MODEL3, TestParameters(TestType.NORMAL, randomSeed = 0))
        val rng1 = Random(1L)
        val rng2 = Random(1L)
        val instances1 = generateSequence {
            bandit1.chooseOrThrow().also {
                bandit1.update(it, TestType.NORMAL.linearRewards(it, rng1))
            }
        }.take(3).toList()
        val instances2 = generateSequence {
            bandit2.chooseOrThrow().also {
                bandit2.update(it, TestType.NORMAL.linearRewards(it, rng2))
            }
        }.take(3).toList()
        for (i in 0 until 3) {
            assertEquals(instances1[i], instances2[i])
        }
        assertContentEquals(instances1, instances2)
    }

    @Suppress("UNCHECKED_CAST")
    @Test
    fun exportIdempotent() {
        for (m in MODELS) {
            val bandit = bandit(m, TestParameters())
            for (i in 0 until 20) {
                val instance = bandit.chooseOrThrow()
                bandit.update(instance, TestType.BINOMIAL.linearRewards(instance, Random))
            }
            val data1 = (bandit as Bandit<BanditData>).exportData()
            val data2 = (bandit as Bandit<BanditData>).exportData()
            when (data1) {
                is InstancesData -> assertContentEquals(data1.instances, (data2 as InstancesData).instances)
                is TreeData -> assertContentEquals(data1.nodes, (data2 as TreeData).nodes)
                is ForestData -> {
                    for (i in data1.trees.indices)
                        assertContentEquals(data1.trees[i], (data2 as ForestData).trees[i])
                }
                is LinearData -> assertContentEquals(data1.weights, (data2 as LinearData).weights)
                else -> throw IllegalArgumentException("Update test with other types")
            }
        }
    }

    @Suppress("UNCHECKED_CAST")
    @Test
    fun exportImport() {
        for (m in MODELS) {
            val parameters = TestParameters()
            val bandit = bandit(m, parameters)
            for (i in 0 until 10) {
                val instance = bandit.chooseOrThrow()
                bandit.update(instance, parameters.type.linearRewards(instance, Random))
            }
            val list1 = (bandit as Bandit<BanditData>).exportData()
            val bandit2 = bandit(m, TestParameters())
            (bandit2 as Bandit<BanditData>).importData(list1)
            assertNotNull(bandit2.choose())
        }
    }

    @Test
    fun relativeWeightImportance() {
        val m = model { nominal("a", 1, 2) }
        val inst0 = BitArray(2, intArrayOf(1))
        val inst1 = BitArray(2, intArrayOf(2))
        for (type in TestType.values()) {
            val rng = Random(0)
            val bandit = bandit(m, TestParameters(type))
            for (i in 0 until 500) {
                val r0 = type.linearRewards(BitArray(2, intArrayOf(0)), rng)
                val r1 = type.linearRewards(BitArray(2, intArrayOf(3)), rng)
                bandit.update(inst0, r0, 0.99f)
                bandit.update(inst0, r1, 0.01f)
                bandit.update(inst1, r0, 0.01f)
                bandit.update(inst1, r1, 0.99f)
            }
            var count = 0
            for (i in 0 until 100)
                if (bandit.chooseOrThrow().isSet(1)) count++
            assertTrue(count > 20, "$type $count")
        }
    }
}

abstract class PredictionBanditTest<B : PredictionBandit<*>> : BanditTest<B>() {

    abstract override fun bandit(model: Model, parameters: TestParameters): B

    @Test
    fun updatePrediction() {
        val m = model { bool();bool() }
        val instances = (0 until 4).asSequence().map { BitArray(2, intArrayOf(it)) }.toList()
        val rng = Random(0)
        for (type in TestType.values()) {
            val bandit = bandit(m, TestParameters(type))
            val means = (0 until 4).asSequence().map { RunningMean() }.toList()
            for (t in 0 until 400) {
                for ((i, inst) in instances.withIndex()) {
                    val result = type.linearRewards(inst, rng)
                    bandit.update(inst, result)
                    means[i].accept(result)
                }
            }
            for (i in 0 until 4)
                assertEquals(means[i].mean, bandit.predict(instances[i]), 0.2f, "$type")
        }
    }
}

data class TestParameters(val type: TestType = TestType.BINOMIAL,
                          val randomSeed: Int = 0,
                          val maximize: Boolean = true,
                          val rewards: DataSample = FullSample()) {

    fun variance(): VarianceFunction {
        return when (type) {
            TestType.BINOMIAL -> BinomialVariance
            TestType.NORMAL -> NormalVariance
            TestType.POISSON -> PoissonVariance
        }
    }

    fun thompsonPolicy(): BanditPolicy {
        return when (type) {
            TestType.BINOMIAL -> ThompsonSampling(BinomialPosterior)
            TestType.NORMAL -> ThompsonSampling(NormalPosterior)
            TestType.POISSON -> ThompsonSampling(PoissonPosterior)
        }
    }

    fun ucbPolicy(): BanditPolicy {
        return when (type) {
            TestType.BINOMIAL -> UCB1Tuned()
            TestType.NORMAL -> UCB1Normal()
            TestType.POISSON -> UCB1Tuned()
        }
    }
}

enum class TestType {

    BINOMIAL {
        override fun linearRewards(mean: Float, trials: Int, rng: Random) = rng.nextBinomial(LogitTransform.apply(mean), trials).toFloat() / trials
    },
    NORMAL {
        override fun linearRewards(mean: Float, trials: Int, rng: Random) = trials.toFloat() * rng.nextNormal(mean, sqrt(1f / trials.toFloat()))
    },
    POISSON {
        override fun linearRewards(mean: Float, trials: Int, rng: Random) = generateSequence { rng.nextPoisson(3f + mean).toFloat() }.take(trials).sum()
    };

    fun linearRewards(instance: Instance, rng: Random): Float {
        val weights = FloatVector(FloatArray(instance.size) { (1 + it) * 0.1f })
        val z = instance dot weights
        val mean = (z - 1f) / (weights.sum() + 1)
        return linearRewards(mean, 1, rng)
    }

    abstract fun linearRewards(mean: Float, trials: Int, rng: Random): Float

    companion object {
        fun random() = values()[Random.nextInt(values().size)]
    }
}
