package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.CompiledDecisionSpace
import com.eignex.combo.decisions.Context
import com.eignex.combo.shared.WitnessTermination
import com.eignex.combo.shared.WitnessValidator
import com.eignex.combo.shared.assignmentKey
import com.eignex.skema.SchemaJson

/** Frozen study input for one suggestion round. Pending outcomes are not training observations. */
class StudySnapshot internal constructor(
    /** Complete immutable observation snapshot. */
    val history: ObservationHistory,
    /** Trial snapshots at the start of the round. */
    val trials: List<Trial>,
    /** Study lifecycle revision used to reject stale reservations. */
    val revision: Long,
    /** Persisted round number for reproducible independent random streams. */
    val round: Long,
    internal val owner: Any,
) {
    /** Outstanding evaluations in this snapshot. */
    val pending: List<Trial> get() = trials.filter { it.status == TrialStatus.Pending }
}

/** Algorithm output, before the study assigns identifiers and commits reservations. */
class ProposalBatch(
    /** Configurations in creation order, without lifecycle IDs. */
    val samples: List<BanditSample>,
    /** Why proposal generation stopped. */
    val termination: WitnessTermination,
)

/** Stateless suggestion algorithm over a frozen study snapshot. Model caches may be retained internally. */
fun interface SuggestionPolicy {
    /** Propose configurations without writing observations or reservations. */
    fun suggest(snapshot: StudySnapshot, count: Int, context: Context): ProposalBatch
}

/**
 * Single-threaded study lifecycle, independent of the surrogate and acquisition strategy.
 * Reservations validate the captured lifecycle and history revisions before committing.
 * Persist both exported study state and observations to retain outstanding reservations.
 *
 * @property space Original Klause decision space.
 * @property store Append-only observation storage.
 */
class Study(val space: CompiledDecisionSpace, val store: ObservationStore = InMemoryObservationStore()) {
    private val identity = Any()
    private val issued = linkedMapOf<Long, Trial>()
    private val validator = WitnessValidator(space)
    private var revision = 0L
    private var nextId = 1L
    private var round = 0L

    /** Current trial snapshots in creation order. */
    val trials: List<Trial> get() = issued.values.toList()

    /** Capture state without advancing the suggestion random stream. */
    fun snapshot(): StudySnapshot = StudySnapshot(store.snapshot(), trials, revision, round, identity)

    /** Advance the reproducible round counter and capture all inputs once. */
    fun beginRound(): StudySnapshot {
        check(round < Long.MAX_VALUE) { "round identifiers exhausted" }
        round++
        revision++
        return snapshot()
    }

    /** Run an algorithm once for the requested batch, then validate and reserve its proposals together. */
    fun suggest(policy: SuggestionPolicy, count: Int = 1, context: Context = Context.Empty): TrialBatch {
        require(count > 0)
        val snapshot = beginRound()
        val proposed = policy.suggest(snapshot, count, context)
        require(proposed.samples.size <= count)
        require(proposed.termination != WitnessTermination.CountReached || proposed.samples.size == count)
        return reserve(snapshot, proposed, context)
    }

    /** Commit a batch only if its snapshot is current and every proposed assignment is feasible and unreserved. */
    fun reserve(snapshot: StudySnapshot, proposals: ProposalBatch, context: Context = Context.Empty): TrialBatch {
        check(
            snapshot.owner === identity && snapshot.revision == revision && snapshot.history.revision == store.revision,
        ) { "stale study snapshot" }
        val keys = trials.filter {
            it.status == TrialStatus.Pending
        }.map { it.sample.sample.assignmentKey() }.toMutableSet()
        for (point in proposals.samples) {
            require(space.matches(point.sample, context) && validator.valid(point.sample)) { "invalid proposal" }
            require(keys.add(point.sample.assignmentKey())) { "proposal is already pending or duplicated" }
        }
        check(nextId <= Long.MAX_VALUE - proposals.samples.size) { "trial identifiers exhausted" }
        val reserved = proposals.samples.map { Trial(nextId++, it, context) }
        for (trial in reserved) issued[trial.id] = trial
        revision++
        return TrialBatch(reserved, proposals.termination)
    }

    /** Append one successful evaluation and close its reservation. */
    fun complete(id: Long, reward: Double, weight: Double = 1.0, noiseVariance: Double? = null) =
        completeWith(id, reward, weight, noiseVariance) { trial, committed ->
            store.append(Observation(trial.sample, trial.context, reward, weight, noiseVariance))
            committed()
        }

    internal fun completeWith(
        id: Long,
        reward: Double,
        weight: Double,
        noiseVariance: Double?,
        record: (Trial, () -> Unit) -> Unit,
    ) {
        val trial = pending(id)
        Observation(trial.sample, trial.context, reward, weight, noiseVariance)
        record(trial) {
            issued[id] = trial.copy(
                status = TrialStatus.Completed,
                reward = reward,
                weight = weight,
                noiseVariance = noiseVariance,
            )
            revision++
        }
    }

    /** Close an unsuccessful evaluation without fabricating an outcome. */
    fun fail(id: Long, reason: String) {
        issued[id] = pending(id).copy(status = TrialStatus.Failed, reason = reason)
        revision++
    }

    /** Release an unevaluated reservation. */
    fun cancel(id: Long, reason: String? = null) {
        issued[id] = pending(id).copy(status = TrialStatus.Cancelled, reason = reason)
        revision++
    }

    private fun pending(id: Long): Trial {
        val trial = requireNotNull(issued[id]) { "unknown trial $id" }
        require(trial.status == TrialStatus.Pending) { "trial $id is already closed" }
        return trial
    }

    /** Capture history, reservations, and random-stream position for transfer. */
    fun exportData(): HistoryData = HistoryData(
        space.definition, store.snapshot(), trials, nextId, round,
        problem = space.problem.takeIf { space.coordinates.schema == null },
    )

    private fun remap(context: Context): Context {
        if (context.bools.keys.all { it in space.contextBools } &&
            context.ints.keys.all { it in space.contextInts } &&
            context.floats.keys.all { it in space.contextFloats }
        ) {
            return context
        }
        val bools = context.bools.entries.associate { it.key.name to it.value }
        val ints = context.ints.entries.associate { it.key.name to it.value }
        val floats = context.floats.entries.associate { it.key.name to it.value }
        return Context(
            space.contextBools.filter { it.name in bools }.associateWith { bools.getValue(it.name) },
            space.contextInts.filter { it.name in ints }.associateWith { ints.getValue(it.name) },
            space.contextFloats.filter { it.name in floats }.associateWith { floats.getValue(it.name) },
            context.nativeAssumptions, context.nativeReals, context.exactInts, context.exactReals,
            context.exactRealValues,
        )
    }

    /** Import compatible original-coordinate state; overlapping reservation identifiers are rejected. */
    fun importData(data: HistoryData) {
        require(
            SchemaJson.encodeToString(com.eignex.combo.decisions.DecisionSpaceDef.serializer(), data.definition) ==
                SchemaJson.encodeToString(com.eignex.combo.decisions.DecisionSpaceDef.serializer(), space.definition),
        ) { "history belongs to a different decision-space definition" }
        require((space.coordinates.schema == null) == (data.problem != null)) {
            "history must use the same native or schema coordinate model"
        }
        if (space.coordinates.schema == null) {
            val original = requireNotNull(data.problem) { "history belongs to a schema model" }
            val codec = com.eignex.klause.formats.json.ProblemJson.default
            require(codec.encodeToString(original) == codec.encodeToString(space.problem)) {
                "history belongs to a different native problem"
            }
        }
        require(data.trials.none { it.id in issued }) { "imported trial identifiers already exist" }
        val observations = data.history.observations.map { it.copy(context = remap(it.context)) }
        val trials = data.trials.map { it.copy(context = remap(it.context)) }
        require(observations.all { space.matches(it.sample.sample, it.context) })
        require(trials.all { space.matches(it.sample.sample, it.context) })
        for (observation in observations) store.append(observation)
        for (trial in trials) issued[trial.id] = trial
        nextId = maxOf(nextId, data.nextTrialId)
        round = maxOf(round, data.round)
        revision++
    }
}
