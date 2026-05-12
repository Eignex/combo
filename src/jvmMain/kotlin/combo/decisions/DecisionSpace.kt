package combo.decisions

import com.eignex.klause.ast.SchemaEntry
import com.eignex.klause.compile.compile
import com.eignex.klause.schema.BoolHandle
import com.eignex.klause.schema.IntHandle
import com.eignex.klause.schema.VariableSchema
import com.eignex.skema.SchemaDef
import kotlin.properties.PropertyDelegateProvider
import kotlin.properties.ReadOnlyProperty

/**
 * Combo-side schema that distinguishes *decision* variables (klause-backed; the bandit
 * chooses their values) from *context* variables (set by the caller at every choose/
 * update call; klause never sees them).
 *
 * Subclasses extend the klause [VariableSchema] declarators (`boolVar`, `intVar`,
 * `nominal`, `floatVar`, `constraint { ... }`) with the new context declarators.
 *
 * Compile the space with [compileSpace] to get a [CompiledDecisionSpace] — a klause
 * `CompiledProblem` plus the feature layout that the linear bandits consume.
 */
abstract class DecisionSpace : VariableSchema() {

    private val _contextBools = mutableListOf<BoolContextHandle>()
    private val _contextInts = mutableListOf<IntContextHandle>()

    internal val contextBools: List<BoolContextHandle> get() = _contextBools
    internal val contextInts: List<IntContextHandle> get() = _contextInts

    protected fun contextBool() =
        PropertyDelegateProvider<DecisionSpace, ReadOnlyProperty<DecisionSpace, BoolContextHandle>> { _, prop ->
            val h = BoolContextHandle(prop.name)
            _contextBools += h
            ReadOnlyProperty { _, _ -> h }
        }

    protected fun contextInt() =
        PropertyDelegateProvider<DecisionSpace, ReadOnlyProperty<DecisionSpace, IntContextHandle>> { _, prop ->
            val h = IntContextHandle(prop.name)
            _contextInts += h
            ReadOnlyProperty { _, _ -> h }
        }

    /** Compile to a structural [CompiledDecisionSpace]. Bandit-specific projections (e.g.
     *  [combo.bandit.glm.LinearFeatureProjection]) layer on top of this. */
    fun compileSpace(): CompiledDecisionSpace {
        val klauseEntries: SchemaDef<SchemaEntry> = this.definition()
        val compiled = this.compile()
        return CompiledDecisionSpace(
            compiled = compiled,
            contextBools = _contextBools.toList(),
            contextInts = _contextInts.toList(),
            schemaDef = klauseEntries,
        )
    }

    /** Cast helper: re-tag a klause [BoolHandle] obtained from `boolVar()`. */
    fun handleOf(klauseHandle: BoolHandle): BoolHandle = klauseHandle

    /** Cast helper: re-tag a klause [IntHandle] obtained from `intVar()`. */
    fun handleOf(klauseHandle: IntHandle): IntHandle = klauseHandle
}
