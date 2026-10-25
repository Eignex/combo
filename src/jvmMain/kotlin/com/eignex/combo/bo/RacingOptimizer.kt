package com.eignex.combo.bo

import com.eignex.combo.decisions.BanditSample
import com.eignex.combo.decisions.Context
import com.eignex.combo.shared.AssignmentKey
import com.eignex.combo.shared.assignmentKey
import com.eignex.koblas.DenseVector
import com.eignex.kumulant.stat.summary.estimateMean
import com.eignex.kumulant.stat.summary.pairedMeanDifference
import kotlin.random.Random

/** Paired mean racing limits; early rejection is a heuristic, not a statistical confidence guarantee. */
data class RacingPolicy(
    /** First challenger comparison size. */
    val minMeasurements: Int = 1,
    /** Geometric increase after a challenger survives a comparison. */
    val growth: Int = 2,
    /** Original-unit disadvantage tolerated before rejecting an incomplete challenger. */
    val rejectionSlack: Double = 0.0,
    /** Maximum simultaneous challenger configurations. */
    val maxActive: Int = 4,
    /** Additional worker attempts after a failed instance/seed measurement. */
    val retries: Int = 0,
    /** Reproducible challenger measurement order, independent of worker completion order. */
    val randomSeed: Int = 0,
    /** Initial incumbent measurements; null completes the full plan before racing. */
    val initialIncumbentMeasurements: Int? = null,
    /** Standard-error allowance before early rejection; zero retains paired-mean rejection. */
    val rejectionStandardErrors: Double = 0.0,
) {
    init {
        require(minMeasurements > 0 && growth >= 2 && maxActive > 0 && retries >= 0)
        require(rejectionSlack >= 0.0 && rejectionSlack.isFinite())
        require(initialIncumbentMeasurements == null || initialIncumbentMeasurements > 0)
        require(rejectionStandardErrors >= 0.0 && rejectionStandardErrors.isFinite())
    }
}

/** Configuration's position in the incumbent race. */
enum class RaceStatus {
    /** Collecting the current paired stage. */
    Active,

    /** Current incumbent evaluated on its completed comparison keys. */
    Incumbent,

    /** Lost a paired comparison or a measurement exhausted its retry allowance. */
    Rejected,

    /** Former incumbent superseded by a paired comparison. */
    Finished,
}

/** Serializable-shaped configuration state exported with worker tasks and BO history. */
data class RaceConfiguration(
    /** BO trial identifier. */
    val id: Long,
    /** Original assignment. */
    val sample: BanditSample,
    /** Current measurement stage size. */
    val target: Int,
    /** Captured race state. */
    val status: RaceStatus,
)

/** Current incumbent with its completed comparison evidence. */
data class RacingIncumbent(
    /** Original configuration. */
    val sample: BanditSample,
    /** Mean over completed search keys or the finished fresh-key comparison. */
    val reward: Double,
    /** Number of paired measurements. */
    val measurements: Int,
)

/** Why an external evaluation batch stopped. */
enum class EvaluationBatchStatus {
    /** Requested number issued. */
    CountReached,

    /** Current stages await outstanding workers. */
    Waiting,

    /** External attempt budget reached. */
    BudgetExhausted,

    /** BO could not find another eligible configuration. */
    SearchExhausted,

    /** Finalists completed fresh-key comparisons or exhausted their retry allowance. */
    ConfirmationComplete,
}

/** Worker requests plus an explicit waiting/budget/search result. */
data class EvaluationBatch(
    /** Newly issued worker snapshots. */
    val tasks: List<EvaluationTask>,
    /** Reason the batch stopped. */
    val status: EvaluationBatchStatus,
)

/** Frozen finalists compared on fresh measurement keys without adapting the shortlist to their outcomes. */
data class RaceConfirmation(
    /** Measurement keys unused during search. */
    val keys: List<EvaluationKey>,
    /** Configuration identifiers selected before observing confirmation results. */
    val configurationIds: List<Long>,
)

/** Transfer state for a fresh compatible racing optimizer; pending workers retain their IDs. */
class RacingData internal constructor(
    /** Underlying original-coordinate observations and configuration reservations. */
    val optimizer: HistoryData,
    /** Fixed paired measurement plan. */
    val keys: List<EvaluationKey>,
    /** Stage and retry rules required when restoring. */
    val policy: RacingPolicy,
    /** Captured configuration stages. */
    val configurations: List<RaceConfiguration>,
    /** All issued task snapshots. */
    val tasks: List<EvaluationTask>,
    /** Closed outcomes, including failures and cancellations. */
    val records: List<EvaluationRecord>,
    /** Confirmed incumbent identifier. */
    val incumbentId: Long?,
    /** Next worker identifier. */
    val nextId: Long,
    /** Optional frozen final comparison. */
    val confirmation: RaceConfirmation? = null,
)

/**
 * Single-threaded intensifier racing fresh BO configurations on matched instance/seed measurements.
 * Initial incumbent effort is configurable. Challengers start with a paired prefix, are rejected when their
 * mean disadvantage exceeds the configured slack and standard-error allowance, or grow their stage.
 * Standard-error rejection is descriptive, not a sequential significance guarantee.
 * Challenger plans use reproducibly shuffled keys to avoid repeatedly favoring one seed or instance.
 * Each successful worker response enters BO history exactly once; failures do not create labels.
 * Instance metadata controls evaluation and pairing, not learned instance-feature marginalization.
 * External workers may complete out of order. Storage backends do not form a distributed transaction.
 *
 * @property optimizer Fresh BO proposal/model owner, configured with [RepeatPolicy.Never].
 * @param keys Fixed instance/seed plan, all at the same optional fidelity.
 * @property policy Stage, concurrency, retry, and early rejection rules.
 * @property context Fixed hard context for every configuration in this race.
 * @property evaluationBudget Total worker attempts, including failures and cancellations.
 * @property store External outcomes, separate from surrogate observations.
 */
class RacingOptimizer(
    val optimizer: BayesianOptimizer,
    keys: List<EvaluationKey>,
    val policy: RacingPolicy = RacingPolicy(),
    val context: Context = Context.Empty,
    val evaluationBudget: Int = Int.MAX_VALUE,
    val store: EvaluationStore = InMemoryEvaluationStore(),
) {
    /** Captured measurement plan, independent of the caller's mutable list. */
    val keys: List<EvaluationKey> = keys.toList()
    private val states = linkedMapOf<Long, RaceConfiguration>()
    private val issued = linkedMapOf<Long, EvaluationTask>()
    private val recorded = linkedMapOf<Long, EvaluationRecord>()
    private val known = mutableSetOf<AssignmentKey>()
    private var incumbentId: Long? = null
    private var nextId = 1L
    private var confirmation: RaceConfirmation? = null
    init {
        require(optimizer.repeatPolicy == RepeatPolicy.Never) { "the intensifier owns repeated measurements" }
        require(optimizer.outcome.noise == ObservationNoise.Noisy) { "racing requires a noisy outcome policy" }
        require(optimizer.trials.isEmpty() && optimizer.store.revision == 0L && store.revision == 0L)
        require(keys.isNotEmpty() && keys.distinct().size == keys.size && policy.minMeasurements <= keys.size)
        require(keys.all { it.budget == keys.first().budget }) { "paired races require one fidelity" }
        require(evaluationBudget > 0)
        require(policy.initialIncumbentMeasurements == null || policy.initialIncumbentMeasurements <= keys.size)
    }

    /** Issued worker snapshots, including closed measurements. */
    val tasks: List<EvaluationTask> get() = issued.values.toList()

    /** Configuration stage snapshots. */
    val configurations: List<RaceConfiguration> get() = states.values.toList()

    /** Incumbent over its completed search keys, or fresh confirmation keys after the final comparison. */
    val incumbent: RacingIncumbent? get() = incumbentId?.let { id ->
        val completed = successes(id).keys
        val plan = confirmation?.takeIf { confirmationDone(it) && it.keys.all { key -> key in completed } }?.keys
            ?: keys.filter { it in completed }
        RacingIncumbent(states.getValue(id).sample, mean(id, plan), plan.size)
    }
    private fun successes(id: Long): Map<EvaluationKey, EvaluationRecord> = recorded.values.filter {
        it.task.configurationId == id && it.task.status == EvaluationStatus.Completed
    }.associateBy { it.task.key }
    private fun mean(id: Long, plan: List<EvaluationKey>): Double {
        val values = successes(id)
        return estimateMean(DenseVector.of(DoubleArray(plan.size) { values.getValue(plan[it]).reward!! })).mean
    }
    private fun plan(state: RaceConfiguration): List<EvaluationKey> = if (state.id == states.keys.first()) {
        keys
    } else {
        keys.shuffled(Random(policy.randomSeed.toLong() xor (state.id * -7046029254386353131L)))
    }
    private fun confirmationFailed(id: Long, value: RaceConfirmation): Boolean = value.keys.any { key ->
        key !in successes(id) && recorded.values.count {
            it.task.configurationId == id && it.task.key == key && it.task.status == EvaluationStatus.Failed
        } > policy.retries
    }
    private fun confirmationDone(value: RaceConfirmation): Boolean = value.configurationIds.all { id ->
        value.keys.all { it in successes(id) } || confirmationFailed(id, value)
    }
    private fun settle() {
        confirmation?.let { value ->
            if (confirmationDone(value)) {
                val eligible = value.configurationIds.filter { id -> value.keys.all { it in successes(id) } }
                val winner = eligible.maxByOrNull { optimizer.outcome.score(mean(it, value.keys)) } ?: return
                incumbentId?.takeIf { it != winner }?.let { id ->
                    states[id] = states.getValue(id).copy(status = RaceStatus.Finished)
                }
                incumbentId = winner
                states[winner] = states.getValue(winner).copy(status = RaceStatus.Incumbent)
            }
            return
        }
        for (state in states.values.toList()) {
            if (state.status != RaceStatus.Active) continue
            if (!plan(state).take(state.target).all { it in successes(state.id) }) continue
            val incumbent = incumbentId
            if (incumbent == null) {
                if (state.target >= (policy.initialIncumbentMeasurements ?: keys.size)) {
                    incumbentId = state.id
                    states[state.id] = state.copy(status = RaceStatus.Incumbent)
                }
                continue
            }
            val plan = plan(state).take(state.target)
            if (!plan.all { it in successes(incumbent) }) continue
            val challengerScore = optimizer.outcome.score(mean(state.id, plan))
            val incumbentScore = optimizer.outcome.score(mean(incumbent, plan))
            val challenger = successes(state.id)
            val incumbentValues = successes(incumbent)
            val difference = pairedMeanDifference(
                DenseVector.of(
                    DoubleArray(plan.size) { optimizer.outcome.score(challenger.getValue(plan[it]).reward!!) },
                ),
                DenseVector.of(
                    DoubleArray(plan.size) { optimizer.outcome.score(incumbentValues.getValue(plan[it]).reward!!) },
                ),
            )
            val clearlyBetter = difference.lowerBound(policy.rejectionStandardErrors) > 0.0
            states[state.id] = when {
                state.target == keys.size || (policy.initialIncumbentMeasurements != null && clearlyBetter) -> {
                    if (challengerScore > incumbentScore) {
                        states[incumbent] = states.getValue(incumbent).copy(status = RaceStatus.Finished)
                        incumbentId = state.id
                        state.copy(status = RaceStatus.Incumbent)
                    } else {
                        state.copy(status = RaceStatus.Rejected)
                    }
                }

                difference.upperBound(policy.rejectionStandardErrors) + policy.rejectionSlack < 0.0 ->
                    state.copy(status = RaceStatus.Rejected)

                else -> state.copy(target = minOf(keys.size.toLong(), state.target.toLong() * policy.growth).toInt())
            }
        }
    }
    private fun missing(state: RaceConfiguration): EvaluationKey? {
        val completed = successes(state.id).keys
        val pending = issued.values.filter {
            it.configurationId == state.id && it.status == EvaluationStatus.Pending
        }.map { it.key }.toSet()
        val required = if (state.id == incumbentId) {
            (
                plan(state).take(state.target) + states.values.filter { it.status == RaceStatus.Active }
                    .flatMap { plan(it).take(it.target) }
                ).distinct()
        } else {
            plan(state).take(state.target)
        }
        return required.firstOrNull { it !in completed && it !in pending }
    }
    private fun fresh(): RaceConfiguration? {
        repeat(16) {
            val trial = optimizer.suggest(1, context).trials.firstOrNull() ?: return null
            if (!known.add(trial.sample.sample.assignmentKey())) {
                optimizer.cancel(trial.id, "configuration already raced")
            } else {
                val target = if (incumbentId ==
                    null
                ) {
                    policy.initialIncumbentMeasurements ?: keys.size
                } else {
                    policy.minMeasurements
                }
                return RaceConfiguration(trial.id, trial.sample, target, RaceStatus.Active).also { states[it.id] = it }
            }
        }
        return null
    }

    /**
     * Freeze the best observed configurations for a final comparison on fresh keys.
     * Call with no outstanding workers and reserve enough remaining budget for every finalist/key.
     * Final recommendation uses only confirmation responses, avoiding reuse of selection outcomes.
     * Confirmation keys must share the search plan's fixed fidelity and remain outside earlier history.
     */
    fun beginConfirmation(freshKeys: List<EvaluationKey>, candidates: Int = 3) {
        require(candidates > 0)
        val finalists = states.values.filter { successes(it.id).isNotEmpty() }
            .sortedByDescending { optimizer.outcome.score(mean(it.id, successes(it.id).keys.toList())) }
            .take(candidates).map { it.id }
        beginConfirmation(freshKeys, finalists)
    }

    /**
     * Freeze explicitly selected evaluated configurations for fresh-key confirmation.
     * Allows a caller to rank finalists by model predictions instead of noisy observed extremes.
     * Only the fresh responses choose the winner; selection scores are never recorded as observations.
     *
     * @param freshKeys Distinct measurement keys unused during search, at the search fidelity.
     * @param configurationIds Distinct raced configuration IDs with successful search measurements.
     */
    fun beginConfirmation(freshKeys: List<EvaluationKey>, configurationIds: List<Long>) {
        require(incumbentId != null) { "establish an incumbent before final confirmation" }
        require(confirmation == null && issued.values.none { it.status == EvaluationStatus.Pending })
        require(freshKeys.isNotEmpty() && freshKeys.distinct().size == freshKeys.size)
        require(freshKeys.none { it in keys } && freshKeys.all { it.budget == keys.first().budget })
        require(configurationIds.isNotEmpty() && configurationIds.distinct().size == configurationIds.size)
        require(configurationIds.all { it in states && successes(it).isNotEmpty() })
        require(configurationIds.size.toLong() * freshKeys.size <= evaluationBudget.toLong() - issued.size)
        confirmation = RaceConfirmation(freshKeys.toList(), configurationIds.toList())
    }

    /** Issue available paired measurements, reserving each configuration/key once while pending. */
    fun ask(count: Int = 1): EvaluationBatch {
        require(count > 0)
        val result = mutableListOf<EvaluationTask>()
        repeat(count) {
            settle()
            if (issued.size >= evaluationBudget) return EvaluationBatch(result, EvaluationBatchStatus.BudgetExhausted)
            confirmation?.let { value ->
                val state = value.configurationIds.firstOrNull { id ->
                    val completed = successes(id).keys
                    val pending = issued.values.filter {
                        it.configurationId == id &&
                            it.status == EvaluationStatus.Pending
                    }.map { it.key }
                    !confirmationFailed(id, value) && value.keys.any { it !in completed && it !in pending }
                }
                if (state == null) {
                    return EvaluationBatch(
                        result,
                        if (confirmationDone(
                                value,
                            )
                        ) {
                            EvaluationBatchStatus.ConfirmationComplete
                        } else {
                            EvaluationBatchStatus.Waiting
                        },
                    )
                }
                val completed = successes(state).keys
                val pending = issued.values.filter {
                    it.configurationId == state &&
                        it.status == EvaluationStatus.Pending
                }.map { it.key }
                val key = value.keys.first { it !in completed && it !in pending }
                check(nextId < Long.MAX_VALUE)
                val task = EvaluationTask(nextId++, state, states.getValue(state).sample, context, key)
                issued[task.id] = task
                result += task
                return@repeat
            }
            var state = states.values.firstOrNull {
                (it.status == RaceStatus.Active || it.status == RaceStatus.Incumbent) && missing(it) != null
            }
            if (state == null) {
                val active = states.values.count { it.status == RaceStatus.Active }
                if ((incumbentId == null && active > 0) || active >= policy.maxActive) {
                    return EvaluationBatch(result, EvaluationBatchStatus.Waiting)
                }
                state = fresh() ?: return EvaluationBatch(
                    result,
                    if (active > 0) {
                        EvaluationBatchStatus.Waiting
                    } else {
                        EvaluationBatchStatus.SearchExhausted
                    },
                )
            }
            check(nextId < Long.MAX_VALUE)
            val task = EvaluationTask(nextId++, state.id, state.sample, context, missing(state)!!)
            issued[task.id] = task
            result += task
        }
        return EvaluationBatch(result, EvaluationBatchStatus.CountReached)
    }
    private fun pending(id: Long): EvaluationTask = requireNotNull(issued[id]).also {
        require(it.status == EvaluationStatus.Pending) { "worker task is already closed" }
    }
    private fun record(value: EvaluationRecord) {
        issued[value.task.id] = value.task
        recorded[value.task.id] = value
        try {
            store.append(value)
        } finally {
            settle()
        }
    }

    /** Commit a successful worker result once, retaining raw outcomes rather than overlapping stage averages. */
    fun complete(id: Long, reward: Double, cost: Double = 1.0, seconds: Double? = null, noiseVariance: Double? = null) {
        val task = pending(id)
        val value = EvaluationRecord(
            task.copy(status = EvaluationStatus.Completed),
            reward,
            cost,
            seconds,
            noiseVariance,
        )
        val before = optimizer.store.revision
        try {
            if (optimizer.trials.single { it.id == task.configurationId }.status == TrialStatus.Pending) {
                optimizer.complete(task.configurationId, reward, noiseVariance = noiseVariance)
            } else {
                optimizer.train(task.sample, context, reward, noiseVariance = noiseVariance)
            }
        } finally {
            if (optimizer.store.revision != before) record(value)
        }
    }

    /** Record unsuccessful work and retry the same paired key only within the configured allowance. */
    fun fail(id: Long, reason: String, cost: Double = 1.0) {
        val task = pending(id)
        val value = EvaluationRecord(task.copy(status = EvaluationStatus.Failed), cost = cost, reason = reason)
        val failures = recorded.values.count {
            it.task.configurationId == task.configurationId && it.task.key == task.key &&
                it.task.status == EvaluationStatus.Failed
        }
        if (failures >= policy.retries && task.key !in (confirmation?.keys ?: emptyList())) {
            states[task.configurationId] = states.getValue(task.configurationId).copy(status = RaceStatus.Rejected)
            if (incumbentId == task.configurationId) incumbentId = null
            if (optimizer.trials.single { it.id == task.configurationId }.status == TrialStatus.Pending) {
                optimizer.fail(task.configurationId, reason)
            }
        }
        record(value)
    }

    /** Release a worker key without inventing a response; a later ask may issue the key again. */
    fun cancel(id: Long, reason: String? = null) {
        val task = pending(id)
        record(EvaluationRecord(task.copy(status = EvaluationStatus.Cancelled), cost = 0.0, reason = reason))
    }

    /** Capture BO observations, measurement identities, pending workers, and race positions together. */
    fun exportData(): RacingData = RacingData(
        optimizer.exportData(),
        keys.toList(),
        policy,
        configurations,
        tasks,
        recorded.values.toList(),
        incumbentId,
        nextId,
        confirmation,
    )

    /** Restore into a fresh compatible race with the same paired plan and hard context. */
    fun importData(data: RacingData) {
        require(issued.isEmpty() && states.isEmpty() && optimizer.store.revision == 0L)
        require(data.keys == keys && data.policy == policy && data.tasks.size <= evaluationBudget)
        require(data.nextId > (data.tasks.maxOfOrNull { it.id } ?: 0L))
        require(data.tasks.map { it.id }.distinct().size == data.tasks.size)
        require(data.configurations.map { it.id }.distinct().size == data.configurations.size)
        val trials = data.optimizer.trials.associateBy { it.id }
        val configurations = data.configurations.associateBy { it.id }
        require(data.incumbentId == data.configurations.singleOrNull { it.status == RaceStatus.Incumbent }?.id)
        require(
            data.configurations.all {
                it.id in trials && it.sample == trials.getValue(it.id).sample && it.target in 1..keys.size &&
                    optimizer.space.matches(it.sample.sample, optimizer.space.assumptionsFor(context))
            },
        )
        require(
            data.tasks.all {
                it.id > 0L && it.configurationId in configurations &&
                    it.sample == configurations.getValue(it.configurationId).sample
            },
        )
        require(data.records.map { it.task.id }.distinct().size == data.records.size)
        val tasks = data.tasks.associateBy { it.id }
        require(data.records.all { it.task == tasks[it.task.id] })
        require(
            data.tasks.all {
                it.key in keys + (data.confirmation?.keys ?: emptyList()) && optimizer.space.matches(
                    it.sample.sample,
                    optimizer.space.assumptionsFor(context),
                )
            },
        )
        require(
            data.records.map { it.task.id }.toSet() == data.tasks.filter {
                it.status != EvaluationStatus.Pending
            }.map { it.id }.toSet(),
        )
        data.confirmation?.let {
            require(it.keys.all { key -> key.budget == keys.first().budget })
            require(
                it.keys.isNotEmpty() && it.keys.distinct().size == it.keys.size && it.keys.none { key -> key in keys },
            )
            require(it.configurationIds.isNotEmpty() && it.configurationIds.distinct().size == it.configurationIds.size)
            require(it.configurationIds.all { id -> id in configurations })
        }
        optimizer.importData(data.optimizer)
        for (state in data.configurations) {
            states[state.id] = state
            known += state.sample.sample.assignmentKey()
        }
        for (task in data.tasks) issued[task.id] = task.copy(context = context)
        for (value in data.records) {
            val rebound = value.copy(task = issued.getValue(value.task.id))
            recorded[rebound.task.id] = rebound
            store.append(rebound)
        }
        incumbentId = data.incumbentId
        nextId = data.nextId
        confirmation = data.confirmation
        settle()
    }
}
