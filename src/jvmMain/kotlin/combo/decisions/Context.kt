package combo.decisions

/**
 * Bag of contextual values supplied to a [combo.bandit.Bandit] at each `choose`/`update`
 * call. Built with [context].
 *
 * Every handle declared on a [DecisionSpace] must be present in the [Context] passed
 * to a [combo.decisions.CompiledDecisionSpace.encode]; missing values raise.
 */
class Context internal constructor(
    internal val bools: Map<BoolContextHandle, Boolean>,
    internal val ints: Map<IntContextHandle, Int>,
) {
    operator fun get(handle: BoolContextHandle): Boolean =
        bools[handle] ?: if (handle.isOptional) false
        else error("missing context value for bool handle '${handle.name}'")

    operator fun get(handle: IntContextHandle): Int =
        ints[handle] ?: if (handle.isOptional) handle.min
        else error("missing context value for int handle '${handle.name}'")

    fun isPresent(handle: BoolContextHandle): Boolean = handle in bools
    fun isPresent(handle: IntContextHandle): Boolean = handle in ints

    companion object {
        val Empty: Context = Context(emptyMap(), emptyMap())
    }
}

class ContextBuilder internal constructor() {
    private val bools = mutableMapOf<BoolContextHandle, Boolean>()
    private val ints = mutableMapOf<IntContextHandle, Int>()

    fun set(handle: BoolContextHandle, value: Boolean) { bools[handle] = value }
    fun set(handle: IntContextHandle, value: Int) { ints[handle] = value }

    internal fun build() = Context(bools.toMap(), ints.toMap())
}

fun context(build: ContextBuilder.() -> Unit): Context =
    ContextBuilder().apply(build).build()
