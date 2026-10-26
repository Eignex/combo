# Scrambled Sobol sampling refinement

Kumulant supplies a general SobolSequence; Combo adds opt-in base-sample selection for qEI,
qNEI, qLogEI, and qLogNEI. Existing constructors and antithetic defaults remain compatible.
The results do not support changing the 256-draw default: frozen ranking gains are modest and
budget-dependent, and fresh whole-loop regret varies across objectives and noise conditions.
No Koblas implementation changes are required.

## Numerical machinery and ownership

Kumulant generates uniform points with the first 1,024
[Joe-Kuo D(6) directions](https://web.maths.unsw.edu.au/~fkuo/sobol/) and LMS plus digital-shift
scrambling, a standard Sobol randomization described by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html).
Normal points use AS 241's inverse CDF on 52-bit cell midpoints. This keeps endpoints finite without
dropping initial points or clipping a continuous normal tail onto an endpoint. Direction-table and
quantile provenance, notices, independent reference tests, and a table regeneration script live in Kumulant.

A seeded engine consumes a fixed randomization stream per coordinate. Coordinate projections and
sample prefixes agree as dimensions and draw counts increase. Combo uses these projections when
adding conditional candidate coordinates, retaining cached baseline draws and factorization work.
Acquisition definitions, pending exclusion, feasible search, and the suggestion lifecycle stay in Combo.
Matrix products and triangular operations continue using Koblas; losses and summaries use Kumulant.

Sobol points are dependent integration points, not IID normal observations. Balanced draw counts
are powers of two. Combo's opt-in policy supports at most 1,024 latent coordinates, including active
baseline and candidate coordinates; this is unrelated to the number of configuration features.
An explicit error directs larger histories to Antithetic; there is no silent fallback or history truncation.

## Frozen acquisition ranking

Reuse all 20 posteriors from the previous follow-up: 12 earlier qEI/qNEI histories, including pending
points, plus eight noisy histories spanning four objectives and two, three, and six feature dimensions.
Each has six candidate coordinates: six singletons, 15 pairs, and 15 quartets. Hold all posterior means,
covariances, baselines, candidate sets, and smoothing parameters fixed. Compare both samplers with
eight streams at 256/1,024/4,096 draws. Kumulant's score stats measure errors against each of the
three archived independent 65,536-draw BoTorch Sobol scrambles.

The table averages assessments across those three references. These repeated assessments are
correlated, not 480 independent posterior experiments. Active cases have reference best utility
above 1e-6; the archive also retains tiny-utility tails and all absolute acquisition errors.
“Within 1%” means selected-set reference utility is within 1% of the reference best over the same
finite candidate sets. Relative regret is 1 - selected/best (using log differences for log acquisition).
References are Monte Carlo estimates and can disagree, especially on rare improvements or near ties.

| Kind | q | Draws | Anti within 1% | Anti mean relative regret | Sobol within 1% | Sobol mean relative regret |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 1 | 256 | 89.1% | 6.92% | 89.1% | 7.02% |
| raw | 1 | 1024 | 93.8% | 3.83% | 93.8% | 3.19% |
| raw | 1 | 4096 | 96.9% | 0.07% | 98.4% | 0.03% |
| raw | 2 | 256 | 67.3% | 11.89% | 67.6% | 11.03% |
| raw | 2 | 1024 | 71.9% | 6.39% | 73.0% | 6.34% |
| raw | 2 | 4096 | 75.8% | 4.08% | 78.1% | 3.94% |
| raw | 4 | 256 | 57.8% | 6.04% | 58.6% | 6.39% |
| raw | 4 | 1024 | 65.9% | 3.17% | 65.1% | 3.77% |
| raw | 4 | 4096 | 71.6% | 3.01% | 75.0% | 2.64% |
| log | 1 | 256 | 90.6% | 5.04% | 89.1% | 6.47% |
| log | 1 | 1024 | 93.8% | 3.64% | 94.5% | 2.22% |
| log | 1 | 4096 | 96.9% | 0.07% | 98.4% | 0.03% |
| log | 2 | 256 | 79.8% | 3.94% | 80.1% | 3.24% |
| log | 2 | 1024 | 77.0% | 2.89% | 79.6% | 1.99% |
| log | 2 | 4096 | 76.8% | 2.40% | 80.6% | 2.03% |
| log | 4 | 256 | 63.0% | 4.06% | 63.5% | 4.34% |
| log | 4 | 1024 | 67.7% | 3.38% | 68.5% | 3.21% |
| log | 4 | 4096 | 73.7% | 2.51% | 77.3% | 2.23% |

At q=4 and 4,096 draws the log-acquisition comparison improves modestly across the three references.
At 256 draws there is no consistent advantage in both ranking metrics. This is evidence about a
fixed finite pool; it is not a convergence guarantee or proof of better optimization trajectories.

The independent captured-normal BoTorch audit adds 432 Sobol candidate sets across q=1/2/4.
Raw values agree within 1e-9 and log values within 1e-8. JVM tests cover both the previous antithetic
fixtures and these Sobol fixtures, plus batch-size growth, reordering, exclusions, and cache reuse.

## Fresh whole-loop validation

Use previously unused seeds 1801–1806 with four objectives, SD 0/.03, q=1/4, and two samplers:
192 runs and 6,144 objective evaluations. Each run has the same 32-evaluation budget, initial design,
exploration probability, 16-point initialization pools, bounded feasible local search, GP fitting
budget, known measurement variances, standardization, and independent recommendation-search budget
as the prior refinement suite. Both methods use qLogNEI with 256 draws. Completions reverse within
each batch; no pending evaluations carry between rounds. Only the normal base-sample policy varies.
Truth scores outcomes and never enters fitting, acquisition scoring, or recommendation selection.
These results were not used to tune sampling parameters or choose a new default.

Each table cell is the median of six paired seeds; smaller regret is better.

| Problem | q | SD | Anti best regret | Sobol best regret | Anti recommendation regret | Sobol recommendation regret |
| --- | --- | --- | --- | --- | --- | --- |
| quadratic2 | 1 | 0.0 | 3.39099e-06 | 1.02088e-06 | 7.01044e-07 | 1.02088e-06 |
| quadratic2 | 1 | 0.03 | 0.000615025 | 0.000408676 | 0.00397096 | 0.00118807 |
| quadratic2 | 4 | 0.0 | 8.42048e-06 | 5.04316e-06 | 3.47307e-06 | 1.79839e-06 |
| quadratic2 | 4 | 0.03 | 0.000337262 | 0.000669574 | 0.00110904 | 0.00183591 |
| branin2 | 1 | 0.0 | 0.000955886 | 0.000724688 | 0.000114182 | 0.000407327 |
| branin2 | 1 | 0.03 | 0.00323912 | 0.00181634 | 0.0118619 | 0.00512725 |
| branin2 | 4 | 0.0 | 0.00144055 | 0.00132829 | 0.00100604 | 0.000632445 |
| branin2 | 4 | 0.03 | 0.00225699 | 0.00284485 | 0.00775923 | 0.0121505 |
| multimodal3 | 1 | 0.0 | 0.0159279 | 0.0171136 | 0.0208541 | 0.0218248 |
| multimodal3 | 1 | 0.03 | 0.0204601 | 0.0175207 | 0.0248049 | 0.0404607 |
| multimodal3 | 4 | 0.0 | 0.0138791 | 0.0138887 | 0.0295833 | 0.0369024 |
| multimodal3 | 4 | 0.03 | 0.0147507 | 0.0197776 | 0.0376084 | 0.0367376 |
| ellipsoid6 | 1 | 0.0 | 0.00247699 | 0.00346238 | 0.00141579 | 0.00196851 |
| ellipsoid6 | 1 | 0.03 | 0.0092429 | 0.00996182 | 0.0151563 | 0.0167358 |
| ellipsoid6 | 4 | 0.0 | 0.00234759 | 0.00347448 | 0.00132465 | 0.00180886 |
| ellipsoid6 | 4 | 0.03 | 0.0103588 | 0.00947048 | 0.0231616 | 0.0156816 |

Paired Sobol wins/losses/ties versus antithetic (absolute tie tolerance 1e-10), 24 pairs per objective:

| Problem | Best evaluated | Recommendation |
| --- | --- | --- |
| quadratic2 | 13/8/3 | 10/12/2 |
| branin2 | 10/11/3 | 8/16/0 |
| multimodal3 | 12/10/2 | 9/15/0 |
| ellipsoid6 | 8/16/0 | 13/11/0 |

There is no uniform end-to-end advantage, so sampling remains opt-in. These runs do not isolate
model error, acquisition-search error, and Monte Carlo error. The prior
[fixed-history recommendation diagnostics](../gp-followups/report.md) treated model and search
separately; a sampling comparison alone does not identify the cause of remaining recommendation errors.

## General normal integration quality and cost

Kumulant's quality runner estimates E[cos(0.2 sum(X))] = exp(-0.02 d) for X ~ N(0, I_d), using
eight independent seeds at each dimension and sample count. Kumulant MeanStat and MseLossStat
compute estimates and RMSE. The pseudo-random reference here is IID, not antithetic. Sobol helps
this integrand in lower dimensions, but the high-dimensional results are mixed.

| Dimensions | Draws | Sampler | RMSE |
| --- | --- | --- | --- |
| 4 | 256 | pseudo | 0.00454528 |
| 4 | 256 | sobol | 0.000994362 |
| 4 | 1024 | pseudo | 0.00249918 |
| 4 | 1024 | sobol | 0.000234246 |
| 4 | 4096 | pseudo | 0.00150325 |
| 4 | 4096 | sobol | 7.57774e-05 |
| 32 | 256 | pseudo | 0.0259759 |
| 32 | 256 | sobol | 0.0167194 |
| 32 | 1024 | pseudo | 0.0168724 |
| 32 | 1024 | sobol | 0.0064002 |
| 32 | 4096 | pseudo | 0.0110805 |
| 32 | 4096 | sobol | 0.0027819 |
| 128 | 256 | pseudo | 0.0568003 |
| 128 | 256 | sobol | 0.0536938 |
| 128 | 1024 | pseudo | 0.0117037 |
| 128 | 1024 | sobol | 0.0252312 |
| 128 | 4096 | pseudo | 0.00786154 |
| 128 | 4096 | sobol | 0.00908498 |

Normal-generation JMH results below use two forks, three one-second warmups, five one-second
measurements per fork, and 256 points per invocation. `draw` reuses an engine and output storage;
`bindAndDraw` also includes a newly seeded engine. Throughput is blocks/second, not points/second.
JMH reports its score error alongside the estimate. The shared Intel i9-12900H host had concurrent
workloads; the large uncertainty on some cells precludes precise general timing claims.
These engine timings exclude GP fitting, factorization, posterior prediction, and acquisition scoring.

| Operation | Dimensions | Sampler | JMH score ± error | Unit |
| --- | --- | --- | --- | --- |
| bindAndDraw | 32 | pseudo | 30235.1 ± 7929.3 | ops/s |
| bindAndDraw | 32 | sobol | 8140.4 ± 197.8 | ops/s |
| bindAndDraw | 128 | pseudo | 4741.6 ± 144.4 | ops/s |
| bindAndDraw | 128 | sobol | 1913.1 ± 22.9 | ops/s |
| draw | 32 | pseudo | 16150.2 ± 195.2 | ops/s |
| draw | 32 | sobol | 12527.4 ± 168.8 | ops/s |
| draw | 128 | pseudo | 4251.5 ± 180.2 | ops/s |
| draw | 128 | sobol | 2144.4 ± 1797.8 | ops/s |

## Using the policy

```kotlin
val acquisition = JointLogNoisyExpectedImprovement(
    sampling = JointNormalSampling.ScrambledSobol,
    samples = 1024,
)
val policy = JointBatchPolicy(acquisition, jointSearch)
```

Existing `JointLogNoisyExpectedImprovement()` retains Antithetic and 256 draws. The same explicit
sampling constructor is available on the other three joint improvement factories. A bound objective
is single-threaded and keeps its normal base samples fixed across candidate scoring.

## Reproduction

```sh
./gradlew refineGpSampling -Pqmc.mode=frozen
./gradlew refineGpSampling -Pqmc.mode=reference
python scripts/audit_gp_reference.py --input build/reports/gp-qmc/frozen.json \
  --output build/reports/gp-qmc/reference.csv
./gradlew refineGpSampling -Pqmc.mode=loop
./gradlew check lintDocs
python scripts/summarize_gp_sampling.py
```

During cross-library development add `-Pkumulant.local` with the Kumulant checkout adjacent to Combo.
The final dependency pins consume the published Kumulant sampler; a sibling checkout is not required.
Compressed CSVs preserve both frozen methods, their Kumulant-scored errors, and every fresh outcome
and trace. The posterior inputs and reference scrambles remain in benchmarks/gp-followups.
Metadata records tested revisions, seeds, and budgets. Copy new outputs into this directory before
regenerating the report. The captured-normal fixtures and independent reference are retained here.

In Kumulant, use the existing benchmark engine:

```sh
./kumulant-bench/benchmark.sh --quality --output /tmp/normal-quality
./kumulant-bench/benchmark.sh --output /tmp/normal-throughput -- \
  '.*NormalSamplingBenchmark.*' -p samples=256 -p dimensions=32,128 \
  -p sampling=pseudo,sobol -f 2 -wi 3 -i 5 -w 1s -r 1s
```
