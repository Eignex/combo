# BO diagnostic campaigns

The `study` parameter selects repository-owned diagnostics in the existing Bench SDK
adapter. Builds, paired placement, retries and retained evidence remain owned by the
lab. These are quality/calibration studies; their descriptive wall times are not
throughput measurements. Results and analyses belong in lab artifacts and issue
receipts, not checked-in result directories.

## Studies and frozen plans

| Study | Arms | Question |
| --- | --- | --- |
| `fitting` | `baseline24,coordinate,mean,prior,multistart,combined` | Fitting-policy ablations and initialization sensitivity (#41) |
| `noise` | `known,unknown,partial` | Scalar noise recovery, latent and future-observation calibration (#51) |
| `constrained` | `coordinate,random` | Exact feasibility, context, reservation and search work (#52) |
| `uncertainty` | `coordinate,grid-map,mixture` | Cost-matched research prototype of parameter uncertainty (#43) |

Every case includes all selected arms for one workload, history/source seed, noise
level, batch size, replicate pattern, pending schedule, context, fit seed and
repetition. Rotate arm order across repetitions. Repetitions reuse seeds and do not
supply independent statistical samples. Keep automatic placement: a complete row
runs on one machine in one worker attempt; different rows and preparation may use
other compatible machines. Replace complete unfinished rows after worker loss and
retain the costs of failed attempts.

`phase=frozen` refits identical prepared random-design histories; `phase=loop` runs
fresh BO with common initial observations and equal external evaluation budgets.
Both retain a separate probe set with independent noise. The initial loop design
contains eight external observations (fewer only when the budget is smaller), with
replicates represented by separate unit-weight rows. Subsequent suggestions use
qLogEI for deterministic cases and qLogNEI for noisy cases, with 16 feasible
candidates, four local steps, six neighbors and two starts. Recommendation search
has 128 candidates, four steps and eight neighbors, independently of suggestion
randomness. These controls are held fixed between arms. This is a new controlled
protocol, not a claim to reproduce the exact historical optimization trajectory.

`phase=historical` instead loads the retained 32-observation validation histories
from commit `3b1b1316c962ea0fd2593482f0ceb79b9aad1fac`. The build task
`:combo-bench:prepareDiagnosticFixtures` downloads the immutable compressed input
and verifies SHA-256 `e51cc99f441a616c1f24272555026ec93383e6aef1fef118fde6c2bf28e29d09`.
No old runner is restored. Select `history-policy=baseline24,combined96` to separate
fitting changes on a fixed history from histories produced by different policies.
Historical seeds are 1901–1906; fresh campaign recipes use disjoint seeds.

Plans freeze the synthetic fixture recipe, exact compressed CSV bytes and their
SHA-256 in each case. Execution decodes those prepared bytes, verifies the frozen
hash after relocation, and retains `prepared-fixture.csv` in its output. Carrying
the bytes in the case also avoids platform-dependent regeneration of noise or
transcendental objective values. Planning
never adds mutable files to the prepared source archive: accepted build/archive
reuse must not depend on which study happened to plan first. The historical gzip
is created by the build task and declared as a SHA-pinned `BenchInput`; workers
verify that existing prepared input after relocation.
The manifest keys planner reuse on the installed jars and pinned history archive, so changing snapshot bytes cannot silently reuse a stale plan. The case also freezes every selection parameter and a digest of the installed JVM
classpath jars. Runtime captures retain individual jar hashes, JVM identity and
Koblas kernel identity. The lab's accepted preparation and execution receipts
supply source/build and physical-worker identities. A mutable Maven snapshot name
alone is not sufficient evidence: preserve the actual prepared bytes and hashes.

## Parameters

The original lightweight suite remains available when `study` is omitted.
Diagnostic requests reject unknown options.

| Parameter | Meaning |
| --- | --- |
| `study` | One study from the table above |
| `phase` | `loop` (default), `frozen`, or `historical` (fitting only) |
| `workloads` | `quadratic2,branin2`, or constrained names below |
| `arms` | Selected arms from the study, all retained in every paired case |
| `seeds` | Comma-separated distinct integer history/optimizer/noise seeds |
| `batch` | `1`, `4`, or both |
| `noise-sd` | Comma-separated nonnegative Gaussian observation-noise deviations |
| `replicates` | Independent evaluations per selected point, 1–4; each costs one external evaluation |
| `pending` | `false,true` schedules; `true` fills two batches before completing the older batch in reverse order |
| `budget` | External evaluations per arm, default 32, maximum 1024; incomplete budgets fail |
| `fit-trials` | Additional covariance candidates, default 96; baseline24 always uses 24 |
| `fit-seed` | Comma-separated fit-stream perturbations; 0 preserves the optimizer stream; use 0–7 for restart sensitivity |
| `grid-nodes` | 3 or 5 nodes per log-parameter axis for uncertainty; point-estimate candidate budget matches the grid size |
| `draws` | Positive even joint acquisition sample count, default 256 |
| `probes` | Held-out predictive probes per row, default 256, maximum 4096 |
| `history-policy` | Historical source policy: `baseline24`, `baseline96`, `combined96`, or a comma-separated selection |
| `context` | Context-log capacity values in [0.1,1]; other workloads use the first selected value |
| `repetitions` | Repeated whole paired rows, default 1 |
| `memory-mib`, `timeout-seconds` | Case resource limits; minimum memory 2048 MiB, default deadline 900 seconds |
| `smoke` | Smaller default budget, probes, draws and seed selection; explicit settings override it |

Known-noise arms receive the oracle observation variance. Unknown arms omit every
row variance; partial arms disclose alternate rows. The unknown-noise GP learns
one scalar variance for the undisclosed rows, not a heteroskedastic function.
Standardization uses Kumulant population variance, matching Combo's response
transform. Known variances scale by the square of that response scale.

## Captures and interpretation

Each case retains original and context-conditioned exact source problem JSON, runtime identity, per-arm original
coordinate/exact witness traces, parameter/jitter/likelihood/penalized-score
snapshots and held-out predictions. Metrics include best-evaluated and recommendation
regret separately, external cost (zero new external calls for frozen/historical refits), retained history size, model/covariance-candidate counts, fitting/search
wall time, RMSE, 95% latent and future-observation coverage, and predictive NLL.
The recommendation-pool mean gap diagnoses search error on the same posterior;
probe truth is used only for scoring, never recommendation selection or labels.

Calibration uses 2,048 fixed, common marginal mixture draws per probe and central
empirical intervals. Future-observation draws add the model's noise estimate (or
the disclosed oracle noise), component by component. This interval Monte Carlo
resolution is explicit; probes within a row are not independent replications.
NLL evaluates component Gaussian densities, with variance floored at 1e-15 and
mixture density floored at 1e-300 to retain finite diagnostics. Kumulant supplies
MSE, summary accumulators and PIT histograms/KS distances. PIT CDFs are estimated from the same common marginal draws; KS distances are descriptive, not independent-probe significance tests. Compare per-seed outcomes and paired differences;
do not select a policy from pooled medians alone.

Constrained workloads are `equality` (x+y=1), `context-log` (logarithmic x and a
pinned real capacity with x+y=capacity), `exact-algebraic` (sqrt(2) context) and
`exact-symbolic` (exp(1) context). They use existing Klause hybrid proposal/repair
paths with independently source-checked feasible seeds. Original-coordinate
checks use Klause's exact source checker, including exact contexts and identity;
normalized feature coordinates are never evidence of feasibility. Retained work
counts cover generated seeds, outer hybrid proposal/backtrack calls and explicit
completed-witness checks, not every validation inside the library. Budget misses
remain exhausted/unknown and fail incomplete-budget cases; they do not establish
infeasibility. Failed and cancelled cases retain partial traces and failure reasons.

## Parameter uncertainty prototype

This implementation exists only in `combo-bench`; production GP interfaces and
fixed-parameter conditioning remain unchanged. Its target is conditional on the
empirically standardized responses and frozen transform, not a fully Bayesian
model of the raw response scaling.

For each covariance parameter theta, let g have a zero-mean Matérn-5/2 prior,
let b ~ N(0,B) independently with B=1 in standardized response units, and set
f(x)=b+g(x). Integrating b gives covariance k_theta(x,x')+B for **every pair**,
not an independent diagonal variance addition. With C=K_theta+noise and
h(x)=1-k_x^T C^-1 1, the equivalent correction to the fixed-zero-mean posterior
is h(x) h(x') / (B^-1+1^T C^-1 1), with the corresponding posterior mean shift.
The code evaluates the equivalent augmented-kernel Gaussian directly using
Kumulant Cholesky and Koblas operations. Its marginal likelihood uses that same
augmented covariance and the proper mean prior.

Each feature length scale has a truncated log-normal prior centered at .5 with
log deviation 1 and range [.125,2]. Signal variance has log center 0, log deviation
2 and range [.25,4]. Unknown noise variance has log center ln(.01), log deviation
2 and range [.0001,1]. The tensor grid uses 3 or 5 equally spaced log nodes and
trapezoid weights. Normalized prior masses define an explicit finite prior;
posterior component weights are proportional to prior mass times marginal
likelihood. This is quadrature research, not optimizer restarts presented as samples.
The truncated support and coarse resolution limit conclusions; check the 5-node
frozen sensitivity before selecting an implementation.

`grid-map` selects the largest posterior-mass component under the same integrated
mean prior, separating covariance integration from the mean-prior change.
`coordinate` uses the existing point-estimate GP with the same number of covariance
candidates as the grid. Candidate counts are equal; acquisition and memory costs
can still differ substantially and are reported separately.

Joint acquisitions average the component EI utilities (using log-sum-exp for the
log acquisitions), with one component governing baseline, pending and candidate
points together. Each component binds frozen common normal samples and its own
baseline cache. Mixture moments are used for prediction diagnostics, never as a
substitute for mixture expected improvement. Function draws choose a component
once; temporary conditioning updates its posterior weight by predictive evidence
and leaves the original model unchanged. Refit caches are scoped to the immutable
observation history; a new full fit rebuilds the standardized grid.

References: [GPML, chapter 2, explicit basis functions](https://gaussianprocess.org/gpml/chapters/RW2.pdf)
and [Snoek et al., Bayesian optimization with parameter integration](https://proceedings.neurips.cc/paper/2012/file/05311655a15b75fab86956663e1819cd-Paper.pdf).
The experiment can validly conclude that this prototype's gain does not justify
shipping it. Keep defaults unless independent fresh seeds support a change.

## Running

Build the launcher, then use `describe`, `plan` and `run` as in the module README.
For historical plans first run `./gradlew :combo-bench:prepareDiagnosticFixtures`.
Committed recipes under `lab/campaigns/` are requests, not result archives. Replace
their `base.ref` with the tested full commit SHA and submit through the maintained
HTTP client's `lab run REQUEST.json`. Keep fetched evidence under ignored build
output. Run fitting/noise development diagnostics before untouched fresh seeds;
evaluate uncertainty after those findings. No throughput claim follows from a
single cold JVM or a small synthetic suite.

Analyze the coordinator's accepted-case snapshot reproducibly:

```sh
lab cases JOB --json > combo-bench/build/JOB-cases.jsonl
"$BENCH" analyze combo-bench/build/JOB-cases.jsonl combo-bench/build/JOB-analysis
```

The analysis retains every paired per-row difference, stratum summaries and
2,000 bootstrap resamples of seed-level averages. It rejects duplicate accepted
case IDs and refuses complete-study claims when any case is failed or unfinished.
Fit-stream perturbations, repeated rows and within-row predictive probes do not
increase the count of independent seeds. Keep the raw coordinator snapshot and
its hash beside the analysis, and retain preparation/failed-attempt costs from the
lab separately from these accepted quality comparisons.

Pass a third analysis argument such as `grid-map` to isolate covariance integration
from the integrated-mean prior in uncertainty comparisons. The default uncertainty
baseline is the existing point-estimate `coordinate` arm.
