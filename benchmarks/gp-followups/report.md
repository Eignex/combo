# GP recommendation, sampling, and allocation follow-ups

The recommendation gap has different causes in these fixtures. More search helps the noiseless
quadratic, but its noisy histories already have small finite-pool posterior-mean search gaps.
A larger GP fitting budget helps the noiseless ellipsoid and can worsen its noisy recommendations.
Joint sampling still has consequential finite-sample ranking errors, especially at q=4.
No recommendation, GP fitting, acquisition, or Monte Carlo defaults change in this refinement.

The production change reuses acquisition scratch buffers and the Kumulant mean accumulator,
precomputes fixed smoothing terms, and uses the exact single-point smooth-maximum identity.
Acquisition definitions remain in Combo. Kumulant supplies losses, summaries, normal sampling,
and Cholesky; Koblas supplies matrix products and triangular operations. Neither dependency changes.

## Frozen recommendation diagnosis

These are discovery diagnostics on the already published 1701–1706 histories, not new validation.
Take all 216 final 32-evaluation histories for quadratic2/ellipsoid6, noise SD 0/.03, q=1/2/4,
and sequential-local/nei-local/lognei-local. Restore reverse completion order within each batch.
The 24-trial standard-search reconstruction matches all original recommendations within 1e-10.
Responses, known observation variances, unit weights, and standardization match the original run.
Truth is used only for evaluation; neither fitting nor recommendation search sees it.

For each fixed history compare GP hyperparameter budgets 24 and 96, and recommendation search:
standard = 128 random candidates, 4 starts, 4 steps, 8 neighbors;
large = 1,024 candidates, 4 starts, 16 steps, 16 neighbors.
Both include evaluated configurations as anchors. Each fit also predicts a shared independent
4,096-point uniform set. The finite-pool gap is recommendation mean minus the minimum mean over
that set, history, and both recommendations; it is not a continuous-global-optimum certificate.
Top RMSE uses the best true decile of that independent set. Kumulant MseLossStat scores errors;
MeanStat scores diagnostic two-standard-deviation latent coverage. CSVs retain likelihood,
length scales, coverage, best true evaluated point, and model-best evaluated point.

Each table cell aggregates 54 histories (six seeds, three batch sizes, three policies).

| Problem | SD | Fit trials | Search | Median true regret | Median pool gap | Top RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| quadratic2 | 0.0 | 24 | standard | 1.65518e-06 | 1.58e-06 | 0.000232 |
| quadratic2 | 0.0 | 24 | large | 9.10711e-08 | 0 | 0.000232 |
| quadratic2 | 0.0 | 96 | standard | 1.73921e-06 | 1.58e-06 | 0.00011 |
| quadratic2 | 0.0 | 96 | large | 8.63119e-08 | 0 | 0.00011 |
| quadratic2 | 0.03 | 24 | standard | 0.00351286 | 2.49e-05 | 0.0134 |
| quadratic2 | 0.03 | 24 | large | 0.00369749 | 0 | 0.0134 |
| quadratic2 | 0.03 | 96 | standard | 0.00309628 | 2.77e-05 | 0.0114 |
| quadratic2 | 0.03 | 96 | large | 0.00287033 | 0 | 0.0114 |
| ellipsoid6 | 0.0 | 24 | standard | 0.00145632 | 0.00056 | 0.0116 |
| ellipsoid6 | 0.0 | 24 | large | 0.00143328 | 0 | 0.0116 |
| ellipsoid6 | 0.0 | 96 | standard | 0.00094125 | 0.000497 | 0.00477 |
| ellipsoid6 | 0.0 | 96 | large | 0.000718554 | 0 | 0.00477 |
| ellipsoid6 | 0.03 | 24 | standard | 0.0138361 | 0.000823 | 0.0304 |
| ellipsoid6 | 0.03 | 24 | large | 0.0138985 | 0 | 0.0304 |
| ellipsoid6 | 0.03 | 96 | standard | 0.0175092 | 0.00136 | 0.0262 |
| ellipsoid6 | 0.03 | 96 | large | 0.0340477 | 0 | 0.0262 |

Increasing search can improve the model score while worsening true regret under noise.
The latent-best evaluated point is an oracle diagnostic, not an implementable recommendation policy;
its advantage does not establish a bug in posterior-mean recommendation. More marginal-likelihood
search is also not uniformly better. These results support retaining configurable independent
recommendation budgets and testing model/prior changes separately on fresh seeds.

## Monte Carlo convergence

Freeze 20 posteriors: the 12 earlier qEI/qNEI fixtures (including pending points in qNEI), plus
24-observation noisy lognei-local histories for four objectives, seeds 1701/1702, q=4.
The latter include two-, three-, and six-dimensional fits. Each posterior has six candidate points;
score all six singletons, 15 pairs, and 15 quartets. Fits, points, and baselines never change.

Use eight antithetic pseudo-random streams at 256, 1,024, and 4,096 draws, plus two independent
65,536-draw streams. These streams are not nested across draw counts. Compare with three independent
65,536-draw BoTorch Sobol scrambles, no baseline pruning, no root caching, double precision.
BoTorch versions are archived. Monte Carlo references are estimates, not exact ground truth.
Kumulant MaeLossStat records acquisition error against each reference; the archived score table also
records selected-set utility, relative regret, ties, and zero fractions.

The following uses the first Sobol scramble (160 selections per row: 20 posteriors × eight streams).
“Within 1%” compares the selected set's reference utility with the best of these same finite sets.
The active subset excludes reference best utility ≤1e-6, explicitly separating tiny-utility tails
from ranking errors of larger magnitude. It has 16 posteriors / 128 selections per row.
Relative regret is 1 − selected/best, using exp(log-selected − log-best) for log acquisition.

| Kind | q | Draws | All within 1% | Active within 1% | Active mean relative regret |
| --- | --- | --- | --- | --- | --- |
| raw | 1 | 256 | 86.2% | 89.1% | 6.89% |
| raw | 1 | 1024 | 90.0% | 93.8% | 3.78% |
| raw | 1 | 4096 | 92.5% | 96.9% | 0.05% |
| raw | 2 | 256 | 66.2% | 70.3% | 10.10% |
| raw | 2 | 1024 | 71.2% | 76.6% | 4.11% |
| raw | 2 | 4096 | 76.2% | 82.0% | 1.45% |
| raw | 4 | 256 | 59.4% | 55.5% | 5.82% |
| raw | 4 | 1024 | 66.9% | 64.8% | 2.77% |
| raw | 4 | 4096 | 71.9% | 71.1% | 2.39% |
| log | 1 | 256 | 88.1% | 90.6% | 4.92% |
| log | 1 | 1024 | 90.0% | 93.8% | 3.53% |
| log | 1 | 4096 | 92.5% | 96.9% | 0.05% |
| log | 2 | 256 | 73.1% | 79.7% | 4.08% |
| log | 2 | 1024 | 73.1% | 78.9% | 2.70% |
| log | 2 | 4096 | 75.6% | 81.2% | 1.56% |
| log | 4 | 256 | 58.8% | 60.2% | 3.88% |
| log | 4 | 1024 | 63.7% | 65.6% | 3.26% |
| log | 4 | 4096 | 66.9% | 71.1% | 1.97% |

Even the large pseudo-random reference is not fully stable. The table below compares the first
65,536-draw antithetic stream against all three Sobol scrambles (60 selections per row).
Rare improvements and near ties can change winners; a relative error near one on utility around
1e-18 has a different practical meaning from the same relative error on a useful acquisition.
These diagnostics do not justify claiming q=4 convergence at 4,096 or changing defaults based solely
on a finite-pool experiment. A variance-reducing sampler and acquisition-ranking stability are
better follow-ups than a blanket draw-count increase; general sampling primitives belong in Kumulant.

| Kind | q | 65,536-draw selections within 1% |
| --- | --- | --- |
| raw | 1 | 93.3% |
| raw | 2 | 95.0% |
| raw | 4 | 80.0% |
| log | 1 | 93.3% |
| log | 2 | 93.3% |
| log | 4 | 80.0% |

The independent captured-normal audit now covers q=4 as well: 432 candidate sets across 12 frozen
posteriors, for both raw and log acquisitions, agree with BoTorch within 1e-9 (raw) and 1e-8 (log). These fixtures are checked in JVM tests,
including alternating batch sizes so reused scratch state cannot leak between evaluations.

## Allocation and timing

JFR allocation samples identified per-draw smoothing arrays and mean accumulators as a large source
of temporary objects. The optimized bound objective owns reusable score/improvement storage and
resets its Kumulant accumulator. For q=1 the smooth maximum is exactly its sole log improvement.
Existing common-normal assignment, reduction order, and single-threaded bound-objective contract remain.

The paired benchmark retains the pre-change acquisition implementation from 53ead84 as a test-only
control. Each independent JVM alternates implementation and raw/log order, uses 500 whole-round
warmups, then measures 30 rounds. Three JVMs give 90 pairs per cell. A round evaluates 20 candidate
sets at 256 draws; fitting and baseline binding are excluded here. Checksums agree within 1e-12.
ThreadMXBean measures current-thread allocation bytes; JFR recordings are separate from timing runs.
The host is an Intel i9-12900H, Temurin 25.0.1, with concurrent workloads. Timings are diagnostic;
allocation reductions are the more reproducible result. No wall-clock regression gate is introduced.

| Baseline | q | Kind | Reference B/eval | Current B/eval | Reduction | Paired median speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 1 | raw | 19827 | 10732 | 45.9% | 1.05× |
| 16 | 1 | log | 46451 | 10732 | 76.9% | 1.62× |
| 16 | 4 | raw | 55304 | 40106 | 27.5% | 1.02× |
| 16 | 4 | log | 88072 | 40106 | 54.5% | 1.27× |
| 64 | 1 | raw | 24003 | 12644 | 47.3% | 1.06× |
| 64 | 1 | log | 50627 | 12644 | 75.0% | 1.40× |
| 64 | 4 | raw | 68376 | 50858 | 25.6% | 1.02× |
| 64 | 4 | log | 101144 | 50858 | 49.7% | 1.22× |
| 128 | 1 | raw | 30707 | 16260 | 47.0% | 1.08× |
| 128 | 1 | log | 57331 | 16260 | 71.6% | 1.30× |
| 128 | 4 | raw | 217944 | 197354 | 9.4% | 1.02× |
| 128 | 4 | log | 250712 | 197354 | 21.3% | 1.16× |

Repeat the full-versus-cached posterior-factor benchmark with 30 warmup rounds, 15 measured rounds,
and three independent JVMs, alternating evaluation order. Each round scores 20 sets at 256 draws;
fitting is excluded and baseline binding included. The same GP moments and antithetic normals feed
both paths; Kumulant MaeLossStat verifies utility agreement below 1e-8. Cached factor orders are
B once per round and q per set, versus B+q per set in the full reference.

| Baseline | q | Full median ms/round | Cached median ms/round | Paired median speedup |
| --- | --- | --- | --- | --- |
| 16 | 1 | 5.178 | 3.463 | 1.56× |
| 16 | 4 | 5.756 | 4.118 | 1.42× |
| 64 | 1 | 21.271 | 4.154 | 4.81× |
| 64 | 4 | 25.806 | 7.035 | 3.71× |
| 128 | 1 | 97.120 | 13.310 | 7.12× |
| 128 | 4 | 53.884 | 8.650 | 5.92× |

Longer warmup removes the previous apparent small-baseline cache regression in these measurements.
There is no evidence here for adding a special small-baseline cutoff.

## Reproduction

Run from the repository root; these optional experiments are not part of the normal test gate:

```sh
./gradlew auditGpFollowups -Pfollowup.mode=recommendation
./gradlew auditGpFollowups -Pfollowup.mode=convergence
# Optional external reference environment: torch, botorch, gpytorch, numpy, scipy.
python scripts/audit_gp_convergence.py
./gradlew auditGpFollowups -Pfollowup.mode=score
for fork in 1 2 3; do
  ./gradlew auditGpFollowups -Pfollowup.mode=allocation -Pfollowup.fork=paired-$fork
  ./gradlew profileGpFactorCache -Pfollowup.fork=$fork
done
./gradlew auditGpAcquisition
python scripts/audit_gp_reference.py --input build/reports/gp-refinement/frozen.json \
  --output build/reports/gp-followups/reference-q4.csv
./gradlew check lintDocs
python scripts/summarize_gp_followups.py
```

Results are written under build/reports/gp-followups and build/reports/gp-refinement.
The checked-in compressed CSVs preserve the measured run; copy new measurements into this directory
before regenerating this report. JFR can be enabled separately with
`-Pfollowup.jfr=/absolute/path/recording.jfr`; its sampled stack summary from the final paired
benchmark is archived in allocation-sites.csv. We group allocation samples by their first Combo
stack frame; sampled weights estimate allocation volume, not retained heap. Use `jfr print --json
--events jdk.ObjectAllocationSample /absolute/path/recording.jfr` to inspect the raw recording.
