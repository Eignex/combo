<p align="center">
  <a href="https://eignex.com/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Eignex/.github/refs/heads/main/profile/banner-white.svg">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Eignex/.github/refs/heads/main/profile/banner.svg">
      <img alt="Eignex" src="https://raw.githubusercontent.com/Eignex/.github/refs/heads/main/profile/banner.svg" style="max-width: 100%; width: 22em;">
    </picture>
  </a>
</p>

# COMBO

Combo optimizes typed Klause configurations under constraints. The library separates
algorithms by how they consume observations:

| Package | Data model | Algorithms |
| --- | --- | --- |
| `com.eignex.combo.bandit` | Streaming updates to model state | Online random forest, linear bandit, univariate bandits |
| `com.eignex.combo.bo` | Retained evaluation history and batch refitting | SMAC-style expected improvement, bootstrap forest Thompson sampling, GP/EI |
| `com.eignex.combo.shared` | Common learner interfaces, feature encoding, feedback, and feasible search | Used by both families |

All public packages use the `com.eignex.combo` prefix. Common learner contracts and
numeric feature encoders live in `com.eignex.combo.shared`.

Streaming learners keep sufficient statistics and incremental tree state. History-based
optimizers keep the evaluations themselves and can reconsider every observation when
rebuilding a model. This data-flow distinction applies whether an application targets
cumulative reward or the best final configuration.

Both families use the same decision-space schema and contextual variables.
[Klause](https://github.com/Eignex/klause) owns constraints and feasible witnesses;
[Kumulant](https://github.com/Eignex/kumulant) supplies streaming models, statistical summaries, posterior
sampling, and shared tree snapshots. Combo owns full-history model fitting and acquisition functions, including
Gaussian and lognormal expected improvement, and their optimization over feasible configurations.

The [Eignex Server](server/README.md) runs Klause, Kumulant, and Combo through one
HTTP API with reusable problem, statistic, and optimizer resources. Start it with
`./gradlew :server:run`.

Using it requires three steps:

1. Declare the decision space with its variables, sub-spaces, and constraints.
2. Pick a streaming learner or history-based optimizer and wire it to the compiled space.
3. Loop choose → serve the configuration → update with the observed reward.

## Decision space

The decision space describes what the optimizer can vary. In the example below the
bandit picks an LLM agent setup (model, temperature, prompt style, and tools) per
request, conditioned on the task type and user tier, and learns from whether the run
succeeded.

```json
{
  "decisionSpace": {
    "name": "AgentPolicy",

    "context": {
      "taskType": { "type": "nominal", "labels": ["coding", "writing", "research", "casual"] },
      "userTier": { "type": "nominal", "labels": ["free", "pro", "enterprise"] }
    },

    "variables": {
      "model":       { "type": "nominal", "labels": ["haiku", "sonnet", "opus"] },
      "temperature": { "type": "float",   "min": 0.0, "max": 1.0 },
      "maxTokens":   { "type": "int",     "min": 256, "max": 8192 },
      "promptStyle": { "type": "nominal", "labels": ["terse", "detailed", "chainOfThought"] },
      "tools": {
        "type": "multiple",
        "labels": ["web_search", "code_exec", "file_read", "bash"]
      }
    },

    "constraints": {
      "freeTierCantUseOpus":  "userTier == \"free\" implies model != \"opus\"",
      "codeExecNeedsBash":    "tools.contains(\"code_exec\") implies tools.contains(\"bash\")",
      "codingTasksNeedTools": "taskType == \"coding\" implies (tools.contains(\"code_exec\") or tools.contains(\"file_read\"))",
      "casualKeepsItCheap":   "taskType == \"casual\" implies (maxTokens <= 1024 and model != \"opus\")"
    }
  }
}
```

Every choose call returns a feasible configuration. Variables are typed: bool, int,
float, nominal (pick one), or multiple (pick any subset).

## Bandit

The flagship is the random forest bandit: an online, constraint-aware contextual
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

End-to-end examples covering cascade-with-backtrack, Thompson sampling on linear
models, and context-conditioned arms live under combo/src/jvmTest/kotlin/com/eignex/combo.

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
respectively, with the observation weight. Use `MaeLossStat()` for mean absolute
error or `MseLossStat()` for mean squared error; callers can provide other paired
scorers appropriate to their model. Off-policy `train` leaves reward and score
tracking untouched. Configuring these scorers can trigger additional fits during updates.

The former `testAbsError` / `trainAbsError` scalar sinks are replaced by these paired
scorers. Replace an absolute-error `MeanStat()` sink with `MaeLossStat()` and read its
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
Joint batch EI is not implemented. Streaming learners retain `choose/update`.

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
outside recorded history. Joint batch EI remains a later acquisition improvement.

See the [GP comparison report](benchmarks/gp-comparison/report.md) for frozen-history
model diagnostics, acquisition-search checks, and equal-budget GP/EI, forest/EI, and
random-search experiments. These comparisons use continuous synthetic spaces.

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

The [refinement study](benchmarks/refinement/report.md) separates frozen-model diagnostics,
acquisition search budgets, initial measurement allocation, guarded racing, and fresh-key
confirmation. Optional leaf-mean variance can overlap tree disagreement and overcover;
response dispersion is not interchangeable with uncertainty about the function.

The [pinned SMAC comparison](benchmarks/smac-comparison.md) runs the actual SMAC 2.4.1
HPO facade alongside this profile and feasible random search. This profile is inspired
by SMAC, with different forest statistics, initialization, search, and race scheduling;
it is not an implementation-equivalent port.

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
provide a separate noiseless reference value. The component tests compare seeded
forest/EI and random search on a bounded continuous quadratic, verify replay from a
fresh study, and check that local acquisition search improves beyond an initial pool.
The [multi-seed benchmark](benchmarks/README.md) additionally compares guided forest/EI,
normalized pool-only forest/EI, a legacy-style pool-only baseline, and random search
on five objectives. It records full regret-versus-cost traces and a checked-in summary.
These synthetic results do not establish universal algorithm superiority.
The [forest/EI investigation](benchmarks/forest-ei-investigation.md) traces repeated
evaluations and acquisition plateaus, with separate ablations and validation quadratics.

## Building

This repository contains two Gradle modules: `:combo` is the published optimization
library, and `:server` is the Eignex Server Ktor application. The root project
aggregates their builds. Source code and ABI references for the library live under
`combo/`; the server depends on the library through `project(":combo")`.

Koblas is a direct public dependency, with platform artifacts pinned independently
of Klause and Kumulant. Vector dot products, scaling, weighted reductions, and batch
matrix construction use Koblas APIs rather than parallel numerical helpers.

The build uses Kbuild 1.3.3, Kotlin 2.4.10, Gradle 9.7.0, and a JDK 25 toolchain
that Gradle provisions automatically. Klause and Kumulant use their latest published
snapshots, including Kumulant’s canonical JSON support merged on October 25, 2026. Exact platform artifact versions are pinned in
`combo/build.gradle.kts`, alongside a shared Koblas version that resolves their conflicting
backend pins. The configured targets are JVM and Linux
x64, matching Klause's available snapshots; Combo's implementation is currently JVM.

Build and validate with the published dependencies:

```sh
./gradlew :combo:jvmTest :server:test
./gradlew check lintDocs
```

For sibling development, `-Pkumulant.local` substitutes `../kumulant`.
`-Pkoblas.local` substitutes `../koblas`; `-Pkoblas.local=../another-checkout` selects an isolated numerical checkout.
Use `-Pdependencies.local` to develop against both `../klause` and `../kumulant`.
The Detekt baseline records existing documentation and style debt surfaced by the
new Kbuild checks; new findings remain checked. ABI references live under `combo/api/`.

JVM applications using the vector backend should enable
`--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED`; the test
tasks already supply these flags.

## Roadmap

- **PGBM**: Probabilistic Gradient Boosting Machines
  ([Sprangers et al., 2021](https://arxiv.org/abs/2106.01682)): boosted trees with
  predictive distributions, slotting in next to the random forest as a stronger
  Thompson-sampling surrogate.
- **Additional history-based surrogates**: extend the shared store/refit lifecycle
  to batch linear models and additional acquisition functions.
- **Joint batching and richer kernels**: joint batch acquisitions and mixed-type GP
  covariance, followed by instance features, censored observations, and multi-fidelity scheduling.
