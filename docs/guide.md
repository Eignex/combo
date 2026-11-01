# Combo library guide

## Decision space

The decision space describes what can vary, which values the caller supplies as
context, and which combinations are allowed. Variables can be Boolean, integer,
continuous, nominal (pick one), or multi-select (pick a subset), with nested and
optional sub-spaces. Context stays fixed during each decision.

For example, this server model tunes a service's worker count, batch size, and
cache setting to minimize measured latency while respecting the current worker
limit:

```json
{
  "space": {
    "variables": {
      "workers": {"int": [1, 32]},
      "batchSize": {"int": [1, 256]},
      "cache": "bool"
    },
    "context": {
      "workerLimit": {"int": [1, 32]}
    },
    "constraints": {
      "capacity": {"le": [{"ref": "workers"}, {"ref": "workerLimit"}]}
    }
  },
  "objective": {"minimize": "latency"},
  "outcome": "latency"
}
```

Constraints use structured expressions in JSON or the typed Kotlin DSL. Proposed
decisions are checked against the original constraints. Bounded search may return
no proposal or an `unknown` result when its budget runs out.

A complete Klause `Problem` can also be used directly, including native factors,
sparse or open domains, and exact constants:

```kotlin
val space = CompiledDecisionSpace.fromProblem(problem)
val projection = SearchTransform(space)
val search = HybridDescentSession.klause(space)
// Use projection and search with the existing BayesianOptimizer.
val optimum = space.minimizeNative(space.coordinates.objective("ints.0"))
```

The native problem is retained unchanged. Source-coordinate names (`bools.0`,
`ints.0`, `reals.0`) and optional schema metadata are kept separately from its
factor representation. Native search delegates source routing to Klause, retaining
open-domain and exact-result semantics. The [HTTP API](../server/README.md#native-problems)
embeds the same canonical document under `problem`.

## Library architecture

[Klause](https://github.com/Eignex/klause) owns constraints and feasible witnesses;
[Kumulant](https://github.com/Eignex/kumulant) supplies streaming models, statistical
summaries, posterior sampling, and shared tree snapshots. Combo composes these with
decision models, full-history model fitting, and acquisition functions, including
Gaussian and lognormal expected improvement and their optimization over feasible
configurations.

The learning algorithms differ in how they consume observations:

| Package | Data model | Algorithms |
| --- | --- | --- |
| `com.eignex.combo.bandit` | Streaming updates to model state | Online random forest, linear bandit, univariate bandits |
| `com.eignex.combo.bo` | Retained evaluation history and batch refitting | SMAC-style expected improvement, bootstrap forest Thompson sampling, GP/EI |
| `com.eignex.combo.shared` | Common learner interfaces, feature encoding, feedback, and feasible search | Used by both families |

All public packages use the `com.eignex.combo` prefix. Streaming learners keep
sufficient statistics and incremental tree state. History-based optimizers retain
the evaluations and can reconsider every observation when rebuilding a model.
Both families use the same decision-space schema and contextual variables.

## Bandit

The random forest bandit is an online, constraint-aware contextual
bandit built on Hoeffding-bound decision trees, each carrying a Bayesian posterior at
every leaf. At decision time it combines candidate sampling with the original
cross-tree decision algorithm, reusing feasible witnesses and asking Klause to repair
or backtrack only when necessary. Context values stay fixed throughout the search.

Exploration is per-leaf Thompson sampling on a Bayesian posterior over the leaf
reward, with online bagging diversifying the trees as updates stream in and mtry
restricting each tree's split candidates to a random subspace. Bools, integers,
nominals, multi-selects, and floats all carry their own typed split kinds,
and every constraint is honored at every choose call by construction, with no
rejection-sample fallback that might quietly violate the model.

The default builder wires the forest to a hybrid Klause search:

```kotlin
val space = MyDecisionSpace().compileSpace()
val bandit = RandomForestBandit.build(
    space = space,
    nbrTrees = 25,
    candidateBatchSize = 8,
    poolCapacity = 64,
    randomSeed = 42,
)
```

Kumulant supplies the growing trees and Normal-Gamma Thompson policy. Each round
samples a batch of feasible candidates and scores their merged leaf posteriors.
Candidates reaching the same leaf tuple share one posterior draw for that round;
adding correlated trees does not multiply the effective observation count.

The original cross-tree descent then pins Boolean and nominal decisions. Compatible
cached witnesses answer those checks without a solver call. A missing witness triggers
a local-search repair seeded from a nearby solution, followed by bounded backtracking
when repair is inconclusive. The best feasible candidate is retained even if a
preferred branch exhausts its search budget. Integer and float splits also affect
candidate scoring. Continuous proposals include solver-produced boundary points and
interior points drawn in the declared parameter geometry. Klause independently checks
each interior proposal against the original problem; a failed proposal retains its
feasible solver seed.

Defaults are 2,000 local-search flips and 10,000 backtracking decisions per request;
pass `localSearch = LocalSearchParams(...)` and `backtrack = BacktrackParams(...)`
to tune them. A search budget miss is `Unknown`, not an infeasibility proof.
`HybridDescentSession.lastResult`, `poolHits`, `proposalCalls`, and `backtrackCalls`
are available through `bandit.descentSession` for inspecting the search. The existing
callback-based builder remains available for other Klause engines and strategies.

The forest's `regression` is Kumulant's `RandomForestRegressionStat`: it owns the
trees, seed streams, online bagging, and immutable statistical snapshots. Combo's
`TreeVectorLayout` supplies a read-only numeric view and maps learned predicates
back to Klause expressions. Numeric coordinates preserve declared units; expression
predicates use Boolean feature slots. Both linear and forest bandits use
`com.eignex.combo.shared` for training, prediction, reward/error tracking, and model
snapshot handling, witness caching, and Klause session adapters. The forest uses
the shared hybrid session; the linear bandit uses its supplied objective optimizer.
Tree and forest `exportData`/`importData` transfer actual Kumulant snapshots;
import merges evidence and requires the same ordered predicate layout.

Then it's a choose → serve → update loop:

```kotlin
val arm = bandit.choose() ?: return     // no feasible witness found within the budgets
serve(arm)                              // hand the configuration to the user
val reward = observe()                  // measure clicks, sales, latency, …
bandit.update(arm, reward)
```

Combo extends Klause's variables with context metadata. Context variables use the
same types and constraints as decision variables; the caller supplies their values,
and each choose call pins them through Klause assumptions. Floats retain Klause's
semantics and decoded values. Combo adds no grid; newly proposed assignments require
an independent Klause feasibility check before use.

Use `bandit.choose(context { ... })`, `bandit.update(arm, context, reward)`, and
`bandit.optimal(context)` for contextual optimization. The same context remains a
hard constraint during candidate generation, branch repair, and solver fallback.

## Numeric search geometry

Numeric variables may declare a scale hint while retaining Klause's original types,
bounds, constraints, and returned values:

```kotlin
import com.eignex.combo.decisions.Scale

val learningRate by floatVar(1e-6, 1.0, scale = Scale.Log)
val workers by intVar(1, 128, scale = Scale.Log)
val dropout by floatVar(0.0, 1.0) // Scale.Linear is the default.
```

Linear proposals allocate equal attention to additive intervals. Log proposals
allocate equal attention to ratios and require positive lower bounds. ReverseLog
uses `-log(1 + max - value)` to concentrate proposals near the upper bound.
Scale hints also apply to `contextInt` and `optionalContextInt`, and survive nested
`DecisionSpaceDef` serialization as local-name `scales` metadata.

Vector feature encoding applies log/reverse-log transforms to numeric values and
interactions, leaving the default linear features in their original units. Online
trees retain source predicates and place default thresholds in the declared geometry.
A linear bandit with nonlinear scale hints ranks shared feasible witness candidates;
it cannot translate those feature weights into a linear objective in source units.
Proposal generation remains bounded, and constrained spaces may return short batches.

## Batch witness search

`com.eignex.combo.shared.WitnessSearch` is the shared candidate-generation boundary:

```kotlin
val candidates = search.witnesses(
    WitnessRequest(
        count = 32,
        assumptions = space.assumptionsFor(context),
        exclude = pendingAssignments,
        maxAttempts = 128,
    ),
    random,
)
```

`WitnessBatch.witnesses` contains distinct feasible assignments. `termination` is
`CountReached`, `BudgetExhausted`, or `Infeasible`; a short batch is not an UNSAT proof.
Excluded assignments are compared by their coordinates, ignoring proof metadata.
External `anchors` are independently verified before cache reuse. `maxAttempts`
bounds proposal and fallback calls together; each engine keeps its configured local
search/backtracking budget. A zero-attempt request can reuse verified cached witnesses.
The default `HybridDescentSession` implements this interface alongside incremental
streaming-tree descent. Optimizers can accept other `WitnessSearch` implementations.

## History-based optimization

```kotlin
import com.eignex.combo.bo.InMemoryObservationStore
import com.eignex.combo.bo.RandomForestOptimizer

val history = InMemoryObservationStore()
val optimizer = RandomForestOptimizer.smac(space, store = history)
// Alternative: RandomForestOptimizer.bootstrapThompson(space, store = history)

val configuration = optimizer.chooseOrThrow()
optimizer.update(configuration, evaluate(configuration))
val recommendation = optimizer.optimalOrThrow()
```

Both algorithms use Combo's batch CART forest fitter, which finds numeric split
thresholds from the observed data. Continuous values retain their declared units.

- **SMAC-style expected improvement** refits a forest when the store revision changes.
  It models the distribution of tree predictions as Gaussian and maximizes expected
  improvement. This implements the forest/EI pattern described in
  [SMAC](https://www.jmlr.org/papers/v23/21-0888.html), using Klause's feasible candidate
  search. The optional intensifier below supplies paired racing; censored trials and
  multi-fidelity scheduling remain unimplemented.
- **Bootstrap Thompson sampling** resamples the full evaluation history with
  replacement and fits a fresh random forest for each exploratory choice. The sampled
  model stays fixed throughout that choice. This follows the bootstrap-model sampling
  approach in [TreeBootstrap](https://www.auai.org/uai2017/proceedings/papers/171.pdf).

`ForestOptimizerConfig` controls the batch forest, initial random observations,
random exploration probability, and EI margin. `optimal` always uses the mean of a
forest fitted to the full history. Contextual choose/update overloads match those of
the streaming forest; observations across contexts train a shared response model.
Historical configurations also enter candidate ranking after Klause checks their
feasibility, preserving observed interior points. Fresh continuous candidates use
scaled, independently checked interior proposals alongside boundary seeds. This is a
bounded candidate search, with no guarantee of global acquisition maximization.

`ObservationStore` is the storage boundary. Its append operation advances a revision,
and `snapshot()` returns a stable, ordered history. The default in-memory store retains
all observations without eviction. A persistent implementation can use the same
interface. Appends must atomically store the row and advance the revision; failed
appends leave both unchanged. Stores belong to one decision-space schema. `train` records off-policy
evaluations without reward/error bookkeeping; zero weight is a no-op and negative
weights are rejected. Import/export transfers schema-bound history snapshots.

Prediction learners accept `testScore` and `trainScore` as Kumulant `PairedStat`
scorers. They receive `(prediction, observed reward)` before and after learning,
respectively, with the observation weight. Use `maeLoss().materialize()` for mean absolute
error or `mseLoss().materialize()` for mean squared error; callers can provide other paired
scorers appropriate to their model. Off-policy `train` leaves reward and score
tracking untouched. Configuring these scorers can trigger additional fits during updates.

The former `testAbsError` / `trainAbsError` scalar sinks are replaced by these paired
scorers. Replace an absolute-error `MeanStat()` sink with `maeLoss().materialize()` and read its
weighted mean and total weight. For a former `SumStat()` sink, the weighted total
is `result.mean * result.totalWeights`. Loss calculation and aggregation belong to Kumulant.

For parallel evaluations, the BO interface follows Vizier's batch suggestion and
pending-trial pattern ([Vizier API](https://oss-vizier.readthedocs.io/en/latest/guides/user/running_vizier.html)):

```kotlin
val batch = optimizer.suggest(count = 8, context = context)
for (trial in batch) {
    // Dispatch trial.sample and retain trial.id for the eventual result.
}
optimizer.complete(batch[0].id, reward = result, weight = 1.0)
optimizer.fail(batch[1].id, reason = "evaluation failed")
optimizer.cancel(batch[2].id)
```

`suggest` returns a `TrialBatch` with a termination status and reserves its points
across later requests. It may return fewer trials than requested. Completed results
may arrive in any order and are recorded once using the trial's captured context.
Pending, failed, and cancelled trials never contribute invented rewards. Failure and
cancellation release a reservation for future suggestions. Zero/nonfinite weights
and repeated or unknown trial completions are rejected. A trial is closed immediately
after its observation commits, even if a later metrics callback fails.

The default batch policy selects sequentially with pending assignments excluded.
`SequentialBatchPolicy(FantasyPolicy.ConstantLiar(value))` and
`SequentialBatchPolicy(FantasyPolicy.PosteriorMean)` add temporary outcomes when
constructing a batch. Conditionable surrogates can condition on those observations;
other surrogates refit temporary datasets. Neither path writes fantasies into the
observation store. Bootstrap Thompson fits a fresh history-bootstrap model per slot.
Opt-in joint batch acquisition is described below. Streaming learners retain `choose/update`.

Bayesian optimizers exclude completed assignments by default for deterministic outcomes.
`RepeatPolicy.NoisyOnly` permits replicates when `OutcomeSpec.noise` is `Noisy`;
`RepeatPolicy.Allow` also supports deliberate deterministic repeats and legacy
`choose/update` loops over finite spaces. `RepeatPolicy.Never` excludes all completed
points. An exhausted suggestion search returns a partial/empty batch with its budget
status rather than silently repeating an evaluation. Recommendations can still return
evaluated points. Forest configuration exposes this policy as `repeatPolicy`.

`Study` owns observations, IDs, reservations, and completion independently of the
suggestion algorithm. A `StudySnapshot` captures history and pending evaluations
once per batch. All proposals are validated before their reservations commit, and
stale or foreign snapshots are rejected. `trials` and `pendingTrials` are immutable
snapshots. `exportData`/`importData` transfers history, reservations, and the algorithm
round counter. Persisting only an `ObservationStore` does not preserve reservations.
Calls remain single-threaded even when external evaluations run concurrently.

## Gaussian-process optimization

For low-dimensional continuous spaces, use the same optimization loop with an exact
Gaussian-process surrogate:

```kotlin
import com.eignex.combo.bo.GaussianProcessSurrogate
import com.eignex.combo.bo.model.GaussianProcessConfig

val optimizer = BayesianOptimizer(
    projection = SearchTransform(space),
    surrogate = GaussianProcessSurrogate(GaussianProcessConfig(learnNoise = true)),
    initialDesign = LatinHypercubeInitialDesign(space, evaluationBudget = 100, seed = 42),
    batchPolicy = SequentialBatchPolicy(FantasyPolicy.PosteriorMean),
    outcome = OutcomeSpec(maximize = false, noise = ObservationNoise.Noisy, standardize = true),
    randomSeed = 42,
)
val trial = optimizer.suggest(1)[0]
optimizer.complete(trial.id, cost, noiseVariance = measurementVariance)
```

Combo owns the batch GP and CART models under `com.eignex.combo.bo.model`; Kumulant
retains online learning, statistical summaries, shared tree snapshots, and Cholesky
factorization. Koblas supplies triangular solves and matrix/vector products.
The GP follows [Gaussian Processes for Machine Learning, chapter 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf).

The default kernel is Matérn five halves; `GaussianKernel.Rbf` selects squared exponential
covariance. Length scales operate in encoded coordinates, so `SearchTransform` supplies
normalized numeric and declared log geometry. Bounded coordinate search maximizes the
marginal likelihood over length scales and signal variance; `learnNoise` also fits noise
for rows whose variance is unspecified. Supplied row variances remain fixed. Set
`hyperparameterTrials = 0` to fix kernel/noise parameters. Mean and variance configuration
uses training-response units, after any requested standardization or response transform.

Optional fitting policies are configured separately, preserving the default empirical mean and
unregularized coordinate search:

```kotlin
import com.eignex.combo.bo.model.GaussianHyperparameterSearch
import com.eignex.combo.bo.model.GaussianMeanEstimation
import com.eignex.combo.bo.model.GaussianProcessFitting
import com.eignex.combo.bo.model.GaussianProcessRegularization

val surrogate = GaussianProcessSurrogate(
    fitting = GaussianProcessFitting(
        meanEstimation = GaussianMeanEstimation.CovarianceWeighted,
        regularization = GaussianProcessRegularization(),
        search = GaussianHyperparameterSearch.Multistart,
        starts = 4,
    ),
    config = GaussianProcessConfig(hyperparameterTrials = 96),
)
```

Covariance-weighted fitting profiles a constant mean at every covariance candidate using
Kumulant's generalized least-squares estimator. An explicit `mean` overrides estimation;
conditioning freezes the estimated mean. The regularizer applies Gaussian penalties to log
covariance parameters, with a dimension-scaled length-scale center. It is a density in log
coordinates, with constants omitted. The score is exposed separately as `logFitScore`;
`logMarginalLikelihood` always reports the likelihood alone. Known row noise is never penalized.
Normalize inputs and standardize responses when using the default regularizer centers.
Multistart search divides the same total candidate budget among seeded starts, including restart
points: fitting evaluates one initial covariance plus `hyperparameterTrials` candidates.
Smaller per-start budgets can weaken refinement. These policies remain opt-in.

Predictions expose latent function uncertainty conditional on fitted parameters, excluding
observation noise and uncertainty about the fitted parameters themselves. Numerical jitter
is separate from noise and reported on `FittedGaussianProcess`. Unit statistical weights
are required: explicit variances and independent replicate rows express measurement evidence.
Exact fitting costs cubic time and quadratic memory in history size. Mixed-type kernels,
censored outcomes, and multi-fidelity scheduling are outside this first GP implementation.

`predictJoint` supplies latent covariance when supported and returns null for marginal-only
models; `predictLogJoint` preserves Gaussian log-response covariance through transforms.
Conditioning freezes fitted parameters and re-factorizes augmented data. Function draws
sample latent points jointly and cache their values across repeated, reordered, and later
queries. First visits to unseen coordinates determine seeded random assignment order;
a draw retains those points and its conditioning cost grows with them. Fantasies remain
outside recorded history. Joint acquisition uses joint moments directly rather than growing a function draw.

The [benchmark suite](../combo-bench/README.md) compares GP/EI, forest/EI, and feasible
random search with equal external evaluation budgets on synthetic spaces.

## Joint batch acquisition

`JointBatchPolicy` optimizes a set of configurations using their joint latent posterior.
`JointExpectedImprovement` (q-EI) uses a fixed incumbent; `JointNoisyExpectedImprovement`
(q-NEI) jointly samples completed configurations and candidates, integrating uncertainty
about the latent incumbent. Pending points enter that sampled baseline, so only the
additional improvement from the new batch contributes. Neither acquisition invents
pending outcomes or adds observation noise to function draws. This follows the joint
Monte Carlo formulation in [BoTorch's acquisition interface](https://botorch.readthedocs.io/en/stable/acquisition.html).

```kotlin
val projection = SearchTransform(space)
val jointSearch = JointLocalAndRandomSearch(projection, witnesses, candidateCount = 32)
val optimizer = BayesianOptimizer(
    projection,
    GaussianProcessSurrogate(),
    initialDesign = LatinHypercubeInitialDesign(space, evaluationBudget = 100, seed = 42),
    batchPolicy = JointBatchPolicy(JointLogNoisyExpectedImprovement(samples = 256), jointSearch),
    outcome = OutcomeSpec(maximize = false, noise = ObservationNoise.Noisy, standardize = true),
    randomSeed = 42,
)
val batch = optimizer.suggest(4)
// Complete each reserved trial once; results can arrive in any order.
```

One frozen snapshot and fitted model serve the entire request. Baseline and pending
coordinates respect the requested hard context, while all pending assignments remain
excluded from proposals. Study validates and reserves the full batch together. Initial
exploration and contexts without a completed baseline use ordinary feasible proposals.
Joint moments are required after warmup; marginal-only forest models fail explicitly.
Bootstrap Thompson sampling is a separate incompatible strategy.

`JointPoolSearch` generates a feasible pool, starts several random sets, and replaces
one member at a time while scoring the entire set. It enforces distinct assignments,
context, exclusions, and Klause feasibility, and can return partial batches on exhaustion.
It searches within its pool. `JointLocalAndRandomSearch` additionally refines every
start with checked normalized neighbors and bounded solver repair, so it can leave
the initial pool. Both searches have bounded budgets and no global-optimum guarantee.
The suggestion trace
counts whole-set acquisition evaluations for joint search; these are more expensive than
single-point scores and should not be compared as equal computational units.

All joint acquisitions approximate joint moments as Gaussian, or sample Gaussian log-response
moments before exponentiating for a log-transformed outcome. Fixed normal base samples
make each bound objective reproducible; candidate row permutations and duplicates
do not change its value. Kumulant supplies normal draws, summaries, and Cholesky factors;
Koblas supplies matrix products. Near-singular joint covariance receives bounded relative
jitter up to `1e-8` only after factorization fails. Exactly deterministic coordinates remain
deterministic. Materially indefinite matrices fail rather than dropping correlations.

`JointLogExpectedImprovement` and `JointLogNoisyExpectedImprovement` implement smoothed
q-LogEI/q-LogNEI. They evaluate improvement directly in log units, with fat-tailed
positive-part and batch-maximum approximations from [Ament et al.](https://arxiv.org/abs/2310.20708).
Taking the logarithm of an already-zero EI cannot recover these scores. The positive-part
`temperature` uses original response units; `batchTemperature` uses log-improvement units.
Both must be positive. Candidate sets containing only baseline coordinates have zero
incremental utility (`-Infinity` in log units), which joint search accepts as a valid score.
These acquisitions are approximate, and temperature and sample count affect the approximation.

Antithetic sampling remains the default. Select Kumulant's scrambled Sobol normal points explicitly:

```kotlin
val acquisition = JointLogNoisyExpectedImprovement(
    sampling = JointNormalSampling.ScrambledSobol,
    samples = 1024,
)
```

The same sampling option is available on all four joint improvement factories. Sobol requires
power-of-two draw counts and supports at most 1,024 latent baseline-plus-candidate coordinates.
Sampler and draw-count choices are explicit; evaluate them for the target workload.

Each bound acquisition caches its baseline posterior work, Cholesky factor, latent draws,
and improvement thresholds. Candidate sets use triangular solves and a conditional covariance
factor of order at most the novel batch size. The baseline factor still costs cubic time once
per round; there is no baseline pruning. The `jointPosterior(baseline)` extension exposes baseline moments and candidate cross-covariance.
The GP advertises `JointPosteriorSurrogate` and caches training-to-baseline solves; models and
decorators without that explicit capability use their own full joint predictions.
`logJointPosterior` and `LogJointPosteriorSurrogate` preserve Gaussian log-response coordinates
through standardization. Opting into a cache requires keeping it consistent with the model's
joint predictions. Bound posterior results are read-only by contract.

GP conditioning, acquisition definitions, and feasible search stay in Combo. Kumulant owns
normal sampling, statistical summaries, and factorization; Koblas supplies matrix products.
The Sobol sequence is a general integration primitive that Kumulant exposes independently of BO.

q-NEI is an improvement criterion, not an information-gain policy for repeated measurements.
All joint policies remain opt-in; sequential batching and model defaults are unchanged.

## SMAC-inspired profile and paired racing

The compact `smac` factory retains its original signed-response defaults. For positive
costs, the budget-aware profile combines ten randomized all-feature batch trees,
log-response EI, a continuous Latin hypercube, 20% independent exploration, and ten
adaptive local-search starts:

```kotlin
val optimizer = RandomForestOptimizer.smacProfile(
    space, evaluationBudget = 100, randomSeed = 42,
)
// Minimize strictly positive costs. For signed responses use responseScale = ResponseScale.Identity.
val trial = optimizer.suggest(1)[0]
optimizer.complete(trial.id, cost)
```

Here `evaluationBudget` sizes the initial design; the caller still controls when to
stop ordinary suggestions. `LatinHypercubeInitialDesign` allocates ten configurations
per independent dimension, capped at 25% of the intended budget. Hard contexts are
pinned, categories select one label, and numeric draws follow the declared scale.
Constrained draws may be rejected and replaced by feasible witnesses, so coverage is
not a uniform constrained Latin hypercube. No variable grid is introduced. Plan slots
count issued configurations, including pending and unsuccessful ones; repeated
measurements of a configuration do not consume extra design slots. An oversized
initial batch finishes with independent feasible random proposals. Exploration bypasses
surrogate fitting and acquisition optimization entirely.

For noisy measurements, a separate `RacingOptimizer` owns external worker requests:

```kotlin
val base = RandomForestOptimizer.smacProfile(
    space, evaluationBudget = 100, randomSeed = 42,
    noise = ObservationNoise.Noisy, repeatPolicy = RepeatPolicy.Never,
)
val race = RacingOptimizer(
    base,
    keys = listOf(11, 23, 37).map { EvaluationKey(instance = "validation", seed = it) },
    evaluationBudget = 100,
)
for (task in race.ask(4).tasks) {
    // Dispatch task.sample with task.key.instance, seed, and optional fixed budget.
    // Later: race.complete(task.id, reward = measuredCost, cost = workerCost).
}
val confirmed = race.incumbent // Null until the first configuration finishes its plan.
```

The first incumbent completes the entire measurement plan. A challenger begins with
one matched instance/seed measurement, then doubles its stage while competitive.
Each challenger has a reproducibly shuffled measurement order to avoid favoring the
same first seed or instance in every early comparison.
Worse paired means trigger early rejection; a fully measured winner becomes the
incumbent. `RacingPolicy` controls stage growth, rejection slack, concurrency, and
retries. `initialIncumbentMeasurements` allows progressive incumbent evaluation; missing
incumbent keys are scheduled before their paired comparisons settle. Setting
`rejectionStandardErrors` requires a mean disadvantage exceeding that descriptive
standard-error allowance. These repeated comparisons have no sequential significance
guarantee, and a single unknown-noise difference cannot justify guarded early rejection.
Each successful worker result trains the surrogate once in raw response units;
stage means never become additional observations. `fail` records unsuccessful cost
without a response, and `cancel` releases the key. Already-issued workers may finish
in any order. Attempt limits include failures and cancellations, and short batches
explicitly distinguish waiting, budget exhaustion, and unsuccessful configuration search.

Initial-design budgets count configurations, while racing spends multiple calls on each.
Use `LatinHypercubeInitialDesign(space, evaluationBudget, measurementsPerConfiguration = 3.0)`
for a three-measurement allowance; this divides the initial configuration cap by three.
The allowance sizes the design and does not enforce an external evaluation budget.
`smacProfile` accepts an `initialDesign` override for composing these policies.

When search ends, `race.beginConfirmation(freshKeys, candidates = 3)` freezes the best
observed configurations and compares them on fresh instance/seed keys. Call after
outstanding workers finish and reserve enough remaining budget for every finalist/key.
The final recommendation uses only those fresh outcomes; they also train the model
once and consume the ordinary external budget. Rejected search configurations can
return to the shortlist. Selection never adapts to confirmation labels. The frozen
shortlist and pending fresh-key workers survive export/import. Permanent failures
exhaust their configured retries without inventing responses; if all finalists fail,
the search incumbent remains the fallback.

`EvaluationStore` persists closed worker outcomes separately from `ObservationStore`.
Use `race.exportData()`/`importData()` with a fresh compatible race to retain pending
workers, IDs, stages, and the paired plan. Restore the same policies and hard context.
Calls remain single-threaded. Independent storage backends are not a distributed
transaction: after a committed model update, a failed evaluation-store append leaves
the worker closed to prevent duplicate training; the exported race state retains its
outcome for recovery. Instances and seeds identify paired measurements; they do not
supply learned instance features. Uneven instance coverage can therefore affect the
pooled surrogate. All keys use one fixed fidelity; budget promotions, censored costs,
and automatic multi-fidelity scheduling are future work.

Optional leaf-mean variance can overlap tree disagreement and overcover; response
dispersion is not interchangeable with uncertainty about the function. This profile
is inspired by SMAC, with different forest statistics, initialization, search, and race
scheduling; it is not an implementation-equivalent port.

## Composing a Bayesian optimizer

`RandomForestOptimizer` is a convenience facade over `BayesianOptimizer`. The components
can also be supplied directly:

```kotlin
val optimizer = BayesianOptimizer(
    projection = SearchTransform(space),
    surrogate = ForestSurrogate(),
    acquisitionFactory = ExpectedImprovement(),
    batchPolicy = SequentialBatchPolicy(FantasyPolicy.PosteriorMean),
    outcome = OutcomeSpec(noise = ObservationNoise.Noisy, standardize = true),
    randomSeed = 42,
)
```

- `Surrogate.fit(TrainingData, random)` produces an immutable `FittedSurrogate`.
  Feature batches are Koblas `DenseMatrix` rows; responses, weights, and known
  observation variances are separate Koblas vectors. Models expose latent moments
  and coherent function draws. Forest variance is ensemble disagreement, not a
  calibrated posterior variance. The default forest mode ignores supplied observation
  noise; `ForestUncertainty.LeafMean` uses the batch model’s optional leaf-mean estimates and
  propagates known variances without treating bootstrap copies as independent evidence. `ConditionableSurrogate` is an optional capability for temporary conditioning.
- `AcquisitionFactory` binds a fixed model, improvement baseline, and outcome policy.
  EI, posterior mean, and coherent posterior Thompson acquisitions use the same
  batch scoring interface. A context with no evaluated baseline uses model-mean
  search rather than inventing an observed reward.
- `AcquisitionOptimizer` controls search independently of scoring. The default
  `LocalAndRandomSearch` starts from feasible candidates and observations, then
  searches neighborhoods of promising points. Klause checks or repairs each move
  under the caller's hard context. Joint real moves interpolate verified seeds with
  identical discrete coordinates, retaining exact rational authority for affine
  equalities. Shared random witness sampling also tries checked joint interpolation
  when coordinate proposals stall, without fitting or scoring a model. Search limits
  bound initial attempts, local
  iterations per start, and neighbors. `LocalSearchPolicy` controls multi-start walks,
  bounded plateau exploration, and shrinking/expanding radii in normalized geometry.
  `LocalSearchPolicy(starts = 1, adaptive = false)` retains the previous fixed-radius
  schedule. This remains a heuristic acquisition maximizer.
  Exact acquisition ties use seeded random ordering to explore tree-model plateaus;
  `AcquisitionTieBreak.First` retains candidate order for reproducing earlier searches.
  Verified excluded anchors still seed local geometry, but cannot be selected for
  evaluation. Each distinct source assignment is scored at most once per search.
- `BatchPolicy` selects all proposals against a frozen study snapshot. `InitialDesign`
  independently controls initial and interleaved random exploration. A custom
  `SuggestionPolicy` can bypass modeling entirely; `StudyRunner` and
  `RandomSuggestionPolicy` provide a feasible random-search baseline.

`SearchTransform` normalizes numeric base coordinates into [0, 1] using declared
scale geometry. Presence and nominal indicators retain their Boolean slots, and
inactive values are masked. It has no interaction columns or linear-model dependency.
Models that need declared products can opt into `InteractionFeatureProjection(SearchTransform(space))`;
this appends interactions to model inputs without changing proposal coordinates.
`decode` accepts only base coordinates and maps them
back into a checked original assignment; invalid categorical combinations, contexts,
and constrained points return null. Constraints are never rewritten into feature space.
Custom feature encoders remain supported and are fixed for an optimizer's lifetime.
For spaces declaring interactions, default BO inputs now have fewer columns. Refit
models trained on the previous expanded vectors, or use `InteractionFeatureProjection`
to retain those products. Exported original-coordinate observation history can be reused.

`OutcomeSpec` centralizes objective direction, response scale, and optional standardization.
`ResponseScale.Log` requires positive observations before recording them. Predictions
use lognormal moments in original units, EI integrates improvement in those units, and
sampled functions exponentiate one coherent log-response draw. Optional standardization
is undone before lognormal prediction or acquisition. Known measurement variances use
a first-order delta approximation under the log transform; they are not exact transformed
noise distributions. A conditionable model receives consistently transformed fantasy
labels and noise. Forest uncertainty remains approximate ensemble disagreement.
Predictions, EI margins, recorded metrics, and fantasy labels stay in original reward
units. `Observation.noiseVariance` / `complete(..., noiseVariance = ...)` describe known
noise separately from statistical weight. Under `ObservationNoise.Noisy`, improvement
baselines use model means at evaluated points, and `bestObserved` aggregates replicate
measurements by weight. `recommendOrThrow` searches the model mean and can recommend
an unevaluated point; `optimalOrThrow` remains its compatibility alias.

`lastSuggestion` reports the study round, frozen history revision, random seed, full
model fit count, and acquisition evaluations. Model fitting is revision-cached with
content-derived seeds; metric predictions do not advance suggestion random streams.
Study round counters survive export. Solver witness caches are ephemeral and are not
exported, so replaying an entire run requires the same initial state and search setup.

`OptimizationBenchmark.run` records best evaluated values, cumulative evaluation cost,
and simple regret when a known optimum is supplied. Synthetic noisy objectives can
provide a separate noiseless reference value.
Run synthetic comparisons through the [lab benchmark suite](../combo-bench/README.md).

## Running the server

Run the HTTP server with `./gradlew :server:run`. It listens on `127.0.0.1:8080`
by default. Set `EIGNEX_SERVER_HOST` and `EIGNEX_SERVER_PORT` to change the address.
Gradle provisions JDK 25 automatically.

For a standalone launcher:

```sh
./gradlew :server:installDist
./server/build/install/eignex-server/bin/eignex-server
```

The launcher requires JDK 25. See the [server guide](../server/README.md) for the
HTTP API and model lifecycle.

JVM applications using the library's vector backend should enable
`--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED`.

## Noisy recommendations

`recommendOrThrow()` reads a frozen history without advancing suggestion randomness or
reserving trials. Repeated calls on unchanged history use the same recommendation seed.
Its model fit is cached; configuring a recommendation search separately lets callers
increase computational effort without increasing the external evaluation budget:

```kotlin
val point = optimizer.recommendOrThrow(
    LocalAndRandomSearch(projection, witnesses, candidateCount = 1024, maxAttempts = 2048),
)
```

Fresh confirmation can use explicitly selected evaluated configuration IDs. For example,
rank `race.configurations` by the optimizer's posterior mean, select a frozen shortlist,
and call `race.beginConfirmation(freshKeys, configurationIds)`. Every selected ID must
have a successful measurement; keys must be fresh and enough worker budget must remain.
The winner uses only confirmation measurements. Confirmation spends external evaluation
budget and remains opt-in. Acquisition, model-fitting, and recommendation budgets
remain independent. Increasing joint acquisition `samples` trades computation for a
more stable finite-sample estimate, without a convergence guarantee.
