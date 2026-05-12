package combo.decisions

import com.eignex.klause.solver.Sample

/**
 * Projects a `(Sample, Context)` pair into the representation [R] that a bandit's
 * machine-learning model consumes. The encoder is the user-pluggable seam between a
 * [CompiledDecisionSpace] and a bandit family.
 *
 * Examples:
 *  - A linear bandit takes [R] = `VectorView`; the encoder lays out a dense feature
 *    vector (raw bool/int values, effect coding, interactions, …).
 *  - A tree bandit takes [R] = a typed row view; the encoder decides whether to expose
 *    `isPresent` indicators for optionals, how to widen nominals, and so on.
 *
 * The choose-path back from a learned model to a klause objective is *not* part of this
 * interface — only some encoders have an invertible representation (linear does, trees
 * don't). Bandit-family-specific subtypes add that capability when they have it.
 */
interface FeatureEncoder<out R> {
    val space: CompiledDecisionSpace

    /** Number of feature columns the encoder produces. Constant for a given encoder. */
    val featureSize: Int

    /** Project (`sample`, `context`) into the model-facing representation. */
    fun encode(sample: Sample, context: Context = Context.Empty): R
}
