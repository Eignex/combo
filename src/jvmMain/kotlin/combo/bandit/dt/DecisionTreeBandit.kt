package combo.bandit.dt

import combo.bandit.ParallelPredictionBandit
import combo.bandit.PredictionBandit
import combo.bandit.PredictionBanditBuilder
import combo.bandit.univariate.BanditPolicy
import combo.bandit.univariate.Greedy
import combo.math.*
import combo.model.Model
import combo.expressions.Variable
import combo.sat.*
import combo.sat.optimizers.LocalSearch
import combo.sat.optimizers.Optimizer
import combo.sat.optimizers.SatObjective
import combo.util.*
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * This bandit uses a univariate bandit algorithm, such as [combo.bandit.univariate.ThompsonSampling]. Each leaf
 * node is a bandit arm.
 *
 * This is a generalization of the following paper: https://arxiv.org/pdf/1706.04687.pdf
 *
 * For more info on delta and tau parameters of the VFDT algorithm check out these resources:
 * https://github.com/ulmangt/vfml/blob/master/weka/src/main/java/weka/classifiers/trees/VFDT.java
 * http://kt.ijs.si/elena_ikonomovska/00-disertation.pdf
 */
class DecisionTreeBandit(val parameters: TreeParameters, root: Node? = null, val allowedVariables: IntArray? = null, seedOffset: Int = 0)
    : PredictionBandit<TreeData>, ITreeParameters by parameters {

    private val randomSequence = RandomSequence(randomSeed + seedOffset)
    private val step = AtomicLong()

    private var nbrNodes = 0
    private var nbrAuditNodes = 0

    var root: Node = if (root != null) {
        // Convert TerminalNodes to AuditNodes as required
        if (root is LeafNode) {
            createNode(EmptyCollection, root.data)
        } else {
            val queue = ArrayQueue<SplitNode>()
            queue.add(root as SplitNode)
            while (queue.size > 0) {
                val r = queue.remove()
                if (r.pos is LeafNode) r.pos = createNode((r.pos as LeafNode).literals, (r.pos as LeafNode).data)
                else queue.add(r.pos as SplitNode)
                if (r.neg is LeafNode) r.neg = createNode((r.neg as LeafNode).literals, (r.neg as LeafNode).data)
                else queue.add(r.neg as SplitNode)
            }
            root
        }
    } else {
        createNode(EmptyCollection, banditPolicy.baseData())
    }
        private set


    override fun chooseOrThrow(assumptions: IntCollection) = opt(false, assumptions)
    override fun optimalOrThrow(assumptions: IntCollection) = opt(true, assumptions)

    override fun predict(instance: Instance) = root.findLeaf(instance).data.mean

    private fun opt(optimal: Boolean, assumptions: IntCollection): Instance {
        var propagated: IntHashSet? = null
        for (restart in 0 until parameters.maxRestarts) {
            if (propagateAssumptions && assumptions.isNotEmpty()) {
                propagated = IntHashSet()
                propagated.addAll(assumptions)
                model.problem.unitPropagation(propagated)
            }
            val node = if (optimal) optNode(propagated ?: assumptions, Greedy, step.get())
            else optNode(propagated ?: assumptions, banditPolicy, step.getAndIncrement())
            val instance = when {
                node == null -> optimizer.witness(propagated ?: assumptions)
                assumptions.isEmpty() -> optimizer.witness(node.literals)
                else -> optimizer.witness((propagated
                        ?: assumptions).mutableCopy(nullValue = 0).apply { addAll(node.literals) })
            }
            if (instance != null)
                return instance
            if (node != null && assumptions.isNotEmpty() && node.blocked != null) {
                node.blocked.put(assumptions)
            }
        }
        throw IterationsReachedException(parameters.maxRestarts)
    }

    private fun optNode(assumptions: IntCollection, policy: BanditPolicy, t: Long): LeafNode? {
        var node = root
        val rng = randomSequence.next()
        while (node is SplitNode) {
            node = if (node.ix.toLiteral(true) in assumptions) node.pos
            else if (node.ix.toLiteral(false) in assumptions) node.neg
            else {
                val p = policy.blank()
                p.addArm(node.neg.data)
                p.addArm(node.pos.data)
                val vNeg = policy.evaluate(node.neg.data, t, maximize, rng)
                val vPos = policy.evaluate(node.pos.data, t, maximize, rng)
                if (vPos > vNeg) node.pos
                else node.neg
            }
        }
        return node as LeafNode
    }

    override fun train(instance: Instance, result: Float, weight: Float) {
        root = root.update(instance, result, weight, banditPolicy)
    }

    private inner class AuditNode(setLiterals: IntCollection, total: VarianceEstimator)
        : LeafNode(setLiterals, total, if (blockQueueSize > 0) RandomListCache(blockQueueSize, randomSeed) else null) {

        var nViewed: Int = 0

        val auditedValues = IntHashSet(nullValue = -1).let { set ->

            val propagatedLiterals = IntHashSet(nullValue = 0)
            propagatedLiterals.addAll(setLiterals)
            model.problem.unitPropagation(propagatedLiterals)

            // Both cases work the same
            // Step 1) loop through variables and add one value per variable
            // Step 2) add values (from allowed variables) until viewedValues is met
            val rng = randomSequence.next()
            if (allowedVariables != null) {
                val myVariables = IntHashSet(nullValue = -1)
                myVariables.addAll(allowedVariables)
                while (myVariables.isNotEmpty() && set.size < viewedValues) {
                    val itr = myVariables.permutation(rng)
                    while (itr.hasNext() && set.size < viewedValues) {
                        val variableIndex = itr.nextInt()
                        val variable = model.index.variable(variableIndex)
                        val splitter = splitters[variable] ?: defaultValueSplitter(model, variable)
                        val valueToSplit = splitter.nextSplit(propagatedLiterals, set, rng)
                        if (valueToSplit >= 0) set.add(valueToSplit)
                        else {
                            myVariables.remove(variableIndex)
                            break
                        }
                    }
                }
            } else {
                val itr = permutation(model.index.nbrVariables, rng).iterator()
                while (itr.hasNext() && set.size < viewedValues) {
                    val variable = model.index.variable(itr.nextInt())
                    val splitter = splitters[variable] ?: defaultValueSplitter(model, variable)
                    val valueToSplit = splitter.nextSplit(propagatedLiterals, set, rng)
                    if (valueToSplit >= 0) set.add(valueToSplit)
                }
                val valueIterator = permutation(model.problem.nbrValues, rng).iterator()
                while (set.size < viewedValues && valueIterator.hasNext()) {
                    val value = valueIterator.nextInt()
                    if (value.toLiteral(true) !in propagatedLiterals && value.toLiteral(false) !in propagatedLiterals)
                        set.add(value)
                }
            }
            set.toArray()
        }

        val dataPos = Array(auditedValues.size) { banditPolicy.baseData() }
        val dataNeg = Array(auditedValues.size) { banditPolicy.baseData() }

        override fun update(instance: Instance, result: Float, weight: Float, banditPolicy: BanditPolicy): Node {
            banditPolicy.accept(data, result, weight)
            nViewed++
            for ((i, ix) in auditedValues.withIndex()) {
                if (instance.isSet(ix)) banditPolicy.accept(dataPos[i], result, weight)
                else {
                    val reifiedLiteral = model.reifiedLiterals[ix]
                    if (!filterMissingData || reifiedLiteral == 0 || instance.literal(reifiedLiteral.toIx()) == reifiedLiteral)
                        banditPolicy.accept(dataNeg[i], result, weight)
                }
            }

            if (nViewed % splitPeriod == 0) {
                val total = dataPos[0].combine(dataNeg[0])
                val (sm1, sm2, bestI) = splitMetric.split(total, dataPos, dataNeg, minSamplesSplit, minSamplesLeaf)
                val eps = hoeffdingBound(delta, data.nbrWeightedSamples, literals.size)
                if (bestI >= 0 && (sm2 / sm1 < 1 - eps || eps < tau)) {

                    val totalPos = dataPos[bestI]
                    val totalNeg = dataNeg[bestI]

                    val posHigh = dataPos[bestI].mean > dataNeg[bestI].mean
                    val inOrder = (posHigh && maximize) || (!posHigh && !maximize)

                    nbrAuditNodes--
                    nbrNodes--

                    val pos: Node
                    val neg: Node
                    val posLiterals = literals.mutableCopy(nullValue = 0).apply { add(auditedValues[bestI].toLiteral(true)) }
                    val negLiterals = literals.mutableCopy(nullValue = 0).apply { add(auditedValues[bestI].toLiteral(false)) }

                    // The order in which the nodes are created give priority to nodes with best score
                    if (inOrder) {
                        pos = createNode(posLiterals, totalPos)
                        neg = createNode(negLiterals, totalNeg)
                    } else {
                        neg = createNode(negLiterals, totalNeg)
                        pos = createNode(posLiterals, totalPos)
                    }
                    return SplitNode(auditedValues[bestI], pos, neg, data)
                }
            }

            return this
        }

        override fun findLeaf(instance: Instance) = this

        private fun hoeffdingBound(delta: Float, count: Float, depth: Int): Float {
            // R = 1 for both binary classification and with variance ratio
            //sqrt(/* R*R* */ ln(1.0f / delta) / (2.0f * count))
            return sqrt((-ln(delta) - depth * ln(deltaDecay)) / 2.0f / count)
        }
    }

    /**
     * Add and index new leaf node. It needs to be added to the correct split node by the caller.
     */
    private fun createNode(setLiterals: IntCollection, total: VarianceEstimator): Node {
        nbrNodes++
        return if (2 + nbrNodes + 2 * nbrAuditNodes <= maxNodes
                && setLiterals.size < model.problem.nbrValues && setLiterals.size < maxDepth) {
            val auditNode: AuditNode
            try {
                auditNode = AuditNode(setLiterals, total)
            } catch (e: UnsatisfiableException) {
                // Can be caused by unit propagation in rare cases, which is fine
                return TerminalNode(setLiterals, total, blockQueueSize, randomSeed)
            }
            val node = when {
                auditNode.auditedValues.size > 1 -> {
                    nbrAuditNodes++
                    auditNode
                }
                auditNode.auditedValues.size == 1 -> {
                    val posLiterals = setLiterals.mutableCopy(nullValue = 0).apply { add(auditNode.auditedValues[0].toLiteral(true)) }
                    val negLiterals = setLiterals.mutableCopy(nullValue = 0).apply { add(auditNode.auditedValues[0].toLiteral(false)) }
                    val pos = createNode(posLiterals, banditPolicy.baseData())
                    val neg = createNode(negLiterals, banditPolicy.baseData())
                    SplitNode(auditNode.auditedValues[0], pos, neg, total)
                }
                else -> TerminalNode(setLiterals, total, blockQueueSize, randomSeed)
            }
            node
        } else {
            TerminalNode(setLiterals, total, blockQueueSize, randomSeed)
        }
    }

    override fun importData(data: TreeData) {
        for (d in data.nodes) {
            root.findLeaves(d.literals).forEach {
                val combined = it.data.combine(d.data)
                it.data = combined
            }
        }
    }

    override fun exportData(): TreeData {
        val data = ArrayList<NodeData>()
        val queue = ArrayList<Pair<IntArray, SplitNode>>()
        if (root is SplitNode) queue.add(EMPTY_INT_ARRAY to root as SplitNode)
        while (queue.isNotEmpty()) {
            val (lits, r) = queue.removeAt(queue.lastIndex)
            val posLits = lits + r.ix.toLiteral(true)
            val negLits = lits + r.ix.toLiteral(false)
            if (r.pos is SplitNode) queue.add(posLits to r.pos as SplitNode)
            else data += NodeData(posLits, (r.pos as LeafNode).data)
            if (r.neg is SplitNode) queue.add(negLits to r.neg as SplitNode)
            else data += NodeData(negLits, (r.neg as LeafNode).data)
        }
        return TreeData(data)
    }

    /**
     * Problem and bandit policy are mandatory
     */
    class Builder(val model: Model, val banditPolicy: BanditPolicy) : PredictionBanditBuilder<TreeData> {

        private var optimizer: Optimizer<SatObjective>? = null
        private var randomSeed: Int = System.currentTimeMillis().toInt()
        private var maximize: Boolean = true
        private var splitMetric: SplitMetric =
                if (banditPolicy.baseData() is BinaryEstimator) GiniCoefficient
                else VarianceReduction
        private var delta: Float = 0.5f
        private var deltaDecay: Float = 0.5f
        private var tau: Float = 0.1f
        private var maxNodes: Int = Int.MAX_VALUE
        private var viewedValues: Int = Int.MAX_VALUE
        private var splitPeriod: Int = 10
        private var minSamplesSplit: Float = banditPolicy.prior.nbrWeightedSamples * 2 + 10.0f
        private var minSamplesLeaf: Float = banditPolicy.prior.nbrWeightedSamples + 4.0f
        private var maxDepth: Int = 50
        private var rewards: DataSample = VoidSample
        private var trainAbsError: DataSample = VoidSample
        private var testAbsError: DataSample = VoidSample
        private var propagateAssumptions: Boolean = true
        private var blockQueueSize: Int = 2
        private var maxRestarts: Int = 5
        private var splitters = HashMap<Variable<*, *>, ValueSplitter>()
        private var filterMissingData: Boolean = true

        private var root: Node? = null

        override fun randomSeed(randomSeed: Int) = apply { this.randomSeed = randomSeed }
        override fun maximize(maximize: Boolean) = apply { this.maximize = maximize }
        override fun rewards(rewards: DataSample) = apply { this.rewards = rewards }
        override fun trainAbsError(trainAbsError: DataSample) = apply { this.trainAbsError = trainAbsError }
        override fun testAbsError(testAbsError: DataSample) = apply { this.testAbsError = testAbsError }

        /** Used to generate complete [Instance] from partial literals at leaf nodes. */
        fun optimizer(optimizer: Optimizer<SatObjective>) = apply { this.optimizer = optimizer }

        @Suppress("UNCHECKED_CAST")
        override fun suggestOptimizer(optimizer: Optimizer<*>) = optimizer(optimizer as Optimizer<SatObjective>)

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

        /** How often we check whether a split can be performed during update. */
        fun splitPeriod(splitPeriod: Int) = apply { this.splitPeriod = splitPeriod }

        /** Minimum number of samples in total of both positive and negative values before a variable can be used for a split. */
        fun minSamplesSplit(minSamplesSplit: Float) = apply { this.minSamplesSplit = minSamplesSplit }

        /** Minimum number of samples of both positive and negative values before a variable can be used for a split. */
        fun minSamplesLeaf(minSamplesLeaf: Float) = apply { this.minSamplesLeaf = minSamplesLeaf }

        /** Maximum depth the tree can grow to. */
        fun maxDepth(maxDepth: Int) = apply { this.maxDepth = maxDepth }

        /** Whether unit propagation before search is performed when assumptions are used. */
        fun propagateAssumptions(propagateAssumptions: Boolean) = apply { this.propagateAssumptions = propagateAssumptions }

        /** When the solver fails to generate an instance with assumptions, the assumptions are added to a blocked circular queue. */
        fun blockQueueSize(blockQueueSize: Int) = apply { this.blockQueueSize = blockQueueSize }

        /** Max restart attempts to [choose] (only relevant with assumptions). */
        fun maxRestarts(maxRestarts: Int) = apply { this.maxRestarts = maxRestarts }

        /** Custom value splitters to use instead of default. */
        fun addSplitter(variable: Variable<*, *>, splitter: ValueSplitter) = apply { this.splitters[variable] = splitter }

        /** Whether data should be automatically filtered for update for variables that are undefined (otherwise counted as false). */
        fun filterMissingData(filterMissingData: Boolean) = apply { this.filterMissingData = filterMissingData }

        override fun importData(data: TreeData) = apply { this.root = data.buildTree(banditPolicy.prior, blockQueueSize, randomSeed) }

        override fun parallel() = ParallelPredictionBandit.Builder(this)

        override fun build(): DecisionTreeBandit {
            val parameters = TreeParameters(model, banditPolicy,
                    optimizer ?: LocalSearch.Builder(model.problem).randomSeed(randomSeed).fallbackCached().build(),
                    randomSeed, maximize, rewards, trainAbsError, testAbsError, splitMetric, delta, deltaDecay,
                    tau, maxNodes, maxDepth, viewedValues, splitPeriod, minSamplesSplit, minSamplesLeaf,
                    propagateAssumptions, splitters, filterMissingData, blockQueueSize, maxRestarts)
            return DecisionTreeBandit(parameters, root, null)
        }
    }
}
