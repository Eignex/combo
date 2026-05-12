package combo.bandit.dt

import combo.bandit.ParallelPredictionBandit
import combo.bandit.PredictionBandit
import combo.bandit.PredictionBanditBuilder
import combo.bandit.univariate.BanditPolicy
import combo.bandit.univariate.Greedy
import combo.bandit.univariate.NormalPosterior
import combo.bandit.univariate.ThompsonSampling
import combo.math.*
import combo.model.Model
import combo.expressions.Root
import combo.expressions.Variable
import combo.sat.*
import combo.sat.optimizers.DeltaLinearObjective
import combo.sat.optimizers.LinearObjective
import combo.sat.optimizers.LocalSearch
import combo.sat.optimizers.Optimizer
import combo.util.*
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.*
import kotlin.random.Random

class RandomForestBandit(val parameters: TreeParameters,
                         val trees: Array<DecisionTreeBandit>,
                         val propagateDecisions: Boolean = true,
                         val instanceSamplingMean: Float = 1f)
    : PredictionBandit<ForestData>, ITreeParameters by parameters {

    private val randomSequence = RandomSequence(randomSeed)
    private val step = AtomicLong(0L)

    override fun predict(instance: Instance): Float {
        var score = 0.0f
        for (t in trees)
            score += t.predict(instance)
        return score / trees.size
    }

    override fun train(instance: Instance, result: Float, weight: Float) {
        val rng = randomSequence.next()
        for (t in trees) {
            val n = rng.nextPoisson(instanceSamplingMean)
            if (n > 0) t.train(instance, result, weight * n)
        }
    }

    override fun optimalOrThrow(assumptions: IntCollection) = opt(true, assumptions)
    override fun chooseOrThrow(assumptions: IntCollection) = opt(false, assumptions)

    private tailrec fun Node.descendTo(decisions: IntCollection): Node {
        if (this is LeafNode) return this
        this as SplitNode
        return when {
            ix.toLiteral(true) in decisions -> pos.descendTo(decisions)
            ix.toLiteral(false) in decisions -> neg.descendTo(decisions)
            else -> this
        }
    }

    private fun opt(optimal: Boolean, assumptions: IntCollection): Instance {
        val allDecisions = IntHashSet(nullValue = 0)
        val decisions = IntArrayList()
        val scores = FloatArrayList()
        allDecisions.addAll(assumptions)
        var problem = if (!propagateDecisions) model.problem
        else Problem(model.problem.nbrValues, model.problem.unitPropagation(allDecisions, true))

        val remainingSplits = HashMap<Int, ArrayList<SplitNode>>(trees.size)
        fun updateSplit(node: Node) {
            val r = node.descendTo(allDecisions)
            if (r is SplitNode)
                remainingSplits.getOrPut(r.ix) { ArrayList() }.add(r)
        }

        trees.forEach { updateSplit(it.root) }

        val t = if (optimal) step.get() else step.getAndIncrement()
        while (remainingSplits.isNotEmpty()) {
            val policy = if (optimal) Greedy else banditPolicy.blank()
            var best = Float.NEGATIVE_INFINITY
            var bestLit = 0

            // 1) select split
            for ((ix, nodes) in remainingSplits) {
                val pos = nodes.asSequence().map { it.pos.data }.reduce { n1, n2 -> n1.combine(n2) }
                pos.updateSampleSize(pos.nbrWeightedSamples / nodes.size)
                val neg = nodes.asSequence().map { it.neg.data }.reduce { n1, n2 -> n1.combine(n2) }
                neg.updateSampleSize(neg.nbrWeightedSamples / nodes.size)

                policy.addArm(pos)
                policy.addArm(neg)

                val posValue = policy.evaluate(pos, t, maximize, Random(t.toInt() xor ix.toLiteral(true)))
                val negValue = policy.evaluate(neg, t, maximize, Random(t.toInt() xor ix.toLiteral(false)))

                if (posValue > best) {
                    bestLit = ix.toLiteral(true)
                    best = posValue
                }
                if (negValue > best) {
                    bestLit = ix.toLiteral(false)
                    best = negValue
                }
            }
            allDecisions.add(bestLit)

            // 2) propagate split
            val propagated: Boolean =
                    if (propagateDecisions) {
                        val s1 = allDecisions.size
                        try {
                            problem = Problem(problem.nbrValues, problem.unitPropagation(allDecisions, true))
                            val s2 = allDecisions.size
                            s1 != s2
                        } catch (e: UnsatisfiableException) {
                            break
                        }
                    } else false

            decisions.add(bestLit)
            scores.add(best)

            // 3) update remainingSplits
            if (propagated) {
                for (ix in remainingSplits.keys.toList()) {
                    if (ix.toLiteral(true) in allDecisions || ix.toLiteral(false) in allDecisions) {
                        remainingSplits.remove(ix)!!.forEach { updateSplit(it) }
                    }
                }
            } else {
                remainingSplits.remove(bestLit.toIx())!!.forEach { updateSplit(it) }
            }
        }

        val finalList = decisions.copy()
        finalList.addAll(assumptions)
        var instance = optimizer.witness(finalList)

        if (instance == null && decisions.isNotEmpty()) {
            // all decisions cannot be satisfied so maximize the number of applicable decisions
            val s = scores.toArray()
            val d = decisions.toArray()
            for (i in d.indices) {
                if (!d[i].toBoolean()) s[i] = -s[i]
                d[i] = d[i].toIx()
            }
            val linearObjective = DeltaLinearObjective(maximize, vectors.sparseVector(problem.nbrValues, s, d))
            @Suppress("UNCHECKED_CAST")
            instance = (optimizer as Optimizer<LinearObjective>).optimizeOrThrow(linearObjective, assumptions)
        }
        return instance ?: optimizer.witnessOrThrow(allDecisions)
    }


    override fun exportData(): ForestData {
        return ForestData(List(trees.size) {
            trees[it].exportData()
        })
    }

    override fun importData(data: ForestData) {
        for (i in 0 until min(data.trees.size, trees.size))
            trees[i].importData(data.trees[i])
    }

    class Builder(val model: Model, val banditPolicy: BanditPolicy = ThompsonSampling(NormalPosterior)) : PredictionBanditBuilder<ForestData> {

        private var trees: Int = 10
        private var optimizer: Optimizer<LinearObjective>? = null
        private var randomSeed: Int = System.currentTimeMillis().toInt()
        private var maximize: Boolean = true
        private var splitMetric: SplitMetric =
                if (banditPolicy.prior is BinaryEstimator) GiniCoefficient
                else VarianceReduction
        private var delta: Float = 0.05f
        private var deltaDecay: Float = 0.5f
        private var tau: Float = 0.1f
        private var maxNodes: Int = Int.MAX_VALUE
        private var viewedValues: Int = Int.MAX_VALUE
        private var viewedVariables: Int =
                if (banditPolicy.baseData() is BinaryEstimator) ceil(sqrt(model.nbrVariables.toFloat())).roundToInt()
                else max(1, ceil(model.nbrVariables / 3f).toInt())
        private var splitPeriod: Int = 10
        private var minSamplesSplit: Float = banditPolicy.baseData().nbrWeightedSamples * 2 + 10.0f
        private var minSamplesLeaf: Float = banditPolicy.baseData().nbrWeightedSamples + 4.0f
        private var maxDepth: Int = 50
        private var rewards: DataSample = VoidSample
        private var trainAbsError: DataSample = VoidSample
        private var testAbsError: DataSample = VoidSample
        private var propagateAssumptions: Boolean = true
        private var splitters = HashMap<Variable<*, *>, ValueSplitter>()
        private var filterMissingData: Boolean = true
        private var importedData: ForestData? = null
        private var instanceSamplingMean: Float = 1f
        private var maxRestarts: Int = 10
        private var propagateDecisions: Boolean = true

        override fun randomSeed(randomSeed: Int) = apply { this.randomSeed = randomSeed }
        override fun maximize(maximize: Boolean) = apply { this.maximize = maximize }
        override fun rewards(rewards: DataSample) = apply { this.rewards = rewards }

        /** How many trees are in the assembly. */
        fun trees(trees: Int) = apply { this.trees = trees }

        /** Used to calculate max set coverage for votes. */
        fun optimizer(optimizer: Optimizer<LinearObjective>) = apply { this.optimizer = optimizer }

        @Suppress("UNCHECKED_CAST")
        override fun suggestOptimizer(optimizer: Optimizer<*>) = optimizer(optimizer as Optimizer<LinearObjective>)

        /** Which split metric to use for deciding what variable to split on. */
        fun splitMetric(splitMetric: SplitMetric) = apply { this.splitMetric = splitMetric }

        /** P-value threshold that variable to split on must overcome relative to second best. Lower value requires more data. */
        fun delta(delta: Float) = apply { this.delta = delta }

        /** [delta] will be multiplied by this once for each split. Used to limit the growth of the tree. */
        fun deltaDecay(deltaDecay: Float) = apply { this.deltaDecay = deltaDecay }

        /** Threshold with which the algorithm splits even if it is not proven best (0 is never, 1 is always) */
        fun tau(tau: Float) = apply { this.tau = tau }

        /** Total number of nodes that are permitted to build. */
        fun maxNodes(maxNodes: Int) = apply { this.maxNodes = maxNodes }

        /** The number of randomly selected values in the variables that leaf nodes consider for splitting the tree further during update. */
        fun viewedValues(viewedValues: Int) = apply { this.viewedValues = viewedValues }

        /** The number of allowed variables per tree in the ensemble. */
        fun viewedVariables(viewedVariables: Int) = apply { this.viewedVariables = viewedVariables }

        /** How often we check whether a split can be performed during update. */
        fun splitPeriod(splitPeriod: Int) = apply { this.splitPeriod = splitPeriod }

        /** Minimum number of samples in total of both positive and negative values before a variable can be used for a split. */
        fun minSamplesSplit(minSamplesSplit: Float) = apply { this.minSamplesSplit = minSamplesSplit }

        /** Minimum number of samples of both positive and negative values before a variable can be used for a split. */
        fun minSamplesLeaf(minSamplesLeaf: Float) = apply { this.minSamplesLeaf = minSamplesLeaf }

        /** Maximum depth the tree can grow to. */
        fun maxDepth(maxDepth: Int) = apply { this.maxDepth = maxDepth }

        /**The total absolute error obtained on a prediction before update. */
        override fun trainAbsError(trainAbsError: DataSample) = apply { this.trainAbsError = trainAbsError }

        /** The total absolute error obtained on a prediction after update. */
        override fun testAbsError(testAbsError: DataSample) = apply { this.testAbsError = testAbsError }

        /** Whether unit propagation before search is performed when assumptions are used. */
        fun propagateAssumptions(propagateAssumptions: Boolean) = apply { this.propagateAssumptions = propagateAssumptions }

        /** Custom value splitters to use instead of default. */
        fun addSplitter(variable: Variable<*, *>, splitter: ValueSplitter) = apply { this.splitters[variable] = splitter }

        /** Whether data should be automatically filtered for update for variables that are undefined (otherwise counted as false). */
        fun filterMissingData(filterMissingData: Boolean) = apply { this.filterMissingData = filterMissingData }

        /** How many times each instance is given to each tree on average (passed to poisson distribution) */
        fun instanceSamplingMean(instanceSamplingMean: Float) = apply { this.instanceSamplingMean = instanceSamplingMean }

        /** Max restart attempts to [choose] (only relevant with assumptions). */
        fun maxRestarts(maxRestarts: Int) = apply { this.maxRestarts = maxRestarts }

        /** Whether decisions should be propagated. */
        fun propagateDecisions(propagateDecisions: Boolean) = apply { this.propagateDecisions = propagateDecisions }

        override fun importData(data: ForestData) = apply { this.importedData = data }

        override fun parallel() = ParallelPredictionBandit.Builder(this)

        private fun sampleVariables(rng: Random, observed: TreeData? = null): IntArray {
            fun variableIndices(variable: Variable<*, *>): IntList {
                var v = variable
                val myList = IntArrayList()
                while (v !is Root) {
                    myList.add(model.index.indexOf(v))
                    v = v.parent.canonicalVariable
                }
                return myList
            }

            val observedVariables = IntHashSet(nullValue = -1)
            if (observed != null) {
                val allSplitValues = IntHashSet(nullValue = -1)
                observed.asSequence().flatMap { node -> node.literals.mapArray { it.toIx() }.asSequence() }.forEach { allSplitValues.add(it) }
                allSplitValues.forEach {
                    for (vi in variableIndices(model.index.variableWithValue(it)))
                        observedVariables.add(vi)
                }
            }
            val perm = permutation(model.nbrVariables, rng)
            for (vi in perm) {
                if (observedVariables.size >= viewedVariables) break
                for (vj in variableIndices(model.index.variable(vi)))
                    observedVariables.add(vj)
            }
            return observedVariables.toArray().apply { sort() }
        }

        override fun build(): RandomForestBandit {
            val treeParameters = TreeParameters(model, banditPolicy,
                    optimizer ?: LocalSearch.Builder(model.problem).randomSeed(randomSeed).fallbackCached().build(),
                    randomSeed, maximize, rewards, trainAbsError, testAbsError, splitMetric, delta, deltaDecay, tau,
                    maxNodes, maxDepth, viewedValues, splitPeriod, minSamplesSplit, minSamplesLeaf,
                    propagateAssumptions, splitters, filterMissingData, 0, maxRestarts)
            val rng = Random(randomSeed)
            val trees = if (importedData != null) {
                Array(importedData!!.size) {
                    val tree = importedData!![it]
                    DecisionTreeBandit(treeParameters, tree.buildTree(banditPolicy.prior, 0, 0), sampleVariables(rng, tree), it)
                }
            } else {
                Array(trees) {
                    DecisionTreeBandit(treeParameters, null, sampleVariables(rng), it)
                }
            }
            return RandomForestBandit(treeParameters, trees, propagateDecisions, instanceSamplingMean)
        }
    }
}
