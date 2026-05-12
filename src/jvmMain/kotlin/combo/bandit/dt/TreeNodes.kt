package combo.bandit.dt

import combo.bandit.univariate.BanditPolicy
import combo.math.VarianceEstimator
import combo.sat.Instance
import combo.sat.not
import combo.sat.toBoolean
import combo.sat.toIx
import combo.util.IntCollection
import combo.util.RandomListCache
import combo.util.isEmpty

sealed class Node(var data: VarianceEstimator) {
    /**
     * Find the exact node that matches the instance. This will always work unless there is an index out of bounds.
     */
    abstract fun findLeaf(instance: Instance): LeafNode

    /**
     * Finds all leaves that match the given literals. This can possibly return all leaves if for example the
     * literals are empty.
     */
    abstract fun findLeaves(setLiterals: IntArray): Sequence<LeafNode>

    abstract fun update(instance: Instance, result: Float, weight: Float, banditPolicy: BanditPolicy): Node

}

class SplitNode(val ix: Int, var pos: Node, var neg: Node, data: VarianceEstimator) : Node(data) {

    override fun update(instance: Instance, result: Float, weight: Float, banditPolicy: BanditPolicy): Node {
        banditPolicy.update(data, result, weight)
        if (instance.isSet(ix)) pos = pos.update(instance, result, weight, banditPolicy)
        else neg = neg.update(instance, result, weight, banditPolicy)
        return this
    }

    override fun findLeaf(instance: Instance) =
            if (instance.isSet(ix)) pos.findLeaf(instance)
            else neg.findLeaf(instance)

    override fun findLeaves(setLiterals: IntArray): Sequence<LeafNode> {
        for (l in setLiterals) {
            if (l.toIx() == ix) {
                return if (l.toBoolean()) pos.findLeaves(setLiterals)
                else neg.findLeaves(setLiterals)
            }
        }
        return pos.findLeaves(setLiterals) + neg.findLeaves(setLiterals)
    }
}

abstract class LeafNode(val literals: IntCollection, data: VarianceEstimator, val blocked: RandomListCache<IntCollection>?) : Node(data) {
    override fun findLeaf(instance: Instance) = this
    override fun findLeaves(setLiterals: IntArray) = sequenceOf(this)

    /**
     * Checks whether the assumptions can be satisfied by the conjunction formed by literals.
     * Both arrays are assumed sorted.
     */
    fun matches(assumptions: IntCollection): Boolean {
        if (assumptions.isEmpty()) return true
        for (lit in literals)
            if (!lit in assumptions) return false
        return true
    }

    /**
     * The block buffer is a set of failed assumptions that should not be immediately tried again. This is applied
     * in the rare case when the assumptions matches the [literals] and the node is chosen and the solver then tries
     * to generate an instance with the given assumption and fails. In that case the assumptions are added to the
     * blocked buffer and a new node is chosen (this node will not be selected due to the assumptions are blocked).
     */
    fun blocks(assumptions: IntCollection): Boolean {
        if (assumptions.isEmpty() || blocked == null) return false
        return blocked.find {
            if (it === assumptions) true
            else {
                var covers = true
                for (lit in it) {
                    if (lit !in assumptions) {
                        covers = false
                        break
                    }
                }
                covers
            }
        } != null
    }
}

class TerminalNode(literals: IntCollection, data: VarianceEstimator, blockQueueSize: Int, randomSeed: Int)
    : LeafNode(literals, data, if (blockQueueSize > 0) RandomListCache(blockQueueSize, randomSeed) else null) {
    override fun update(instance: Instance, result: Float, weight: Float, banditPolicy: BanditPolicy) =
            this.apply { banditPolicy.update(data, result, weight) }
}
