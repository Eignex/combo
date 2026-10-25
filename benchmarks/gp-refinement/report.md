# GP acquisition, search, and posterior-cache refinement

This cycle isolates acquisition correctness, sampling/search behavior, and baseline reuse. It keeps GP kernels, likelihood fitting, sequential batching, and exploration defaults unchanged. New log-improvement acquisitions and joint local search remain opt-in.

## Frozen-posterior diagnosis

Twelve frozen cases use the first 24 observations of the archived exclusion/q-NEI histories for discovery seeds 1501–1503 at noise SD .03. Six q-EI cases have a fixed incumbent; six q-NEI cases include one additional pending point in the latent baseline. Each case enumerates six singleton candidates and fifteen pairs.

BoTorch receives exactly the same posterior means, covariance, candidate/baseline rows, and 256 antithetic normal draws. All 252 legacy EI values agree within `1e-9`; their mean absolute differences are around floating-point rounding. Cached EI preserves that agreement, and the new log acquisitions agree with BoTorch log acquisition within `1e-8` on the same draws. These checks isolate acquisition and sampling; they do not compare GP fitting or hyperparameter learning.

An independent 16,384-draw Sobol reference changes the winning pair in 2/6 q-EI cases and 0/6 q-NEI cases. Many 256-draw EI estimates are exactly zero because no sampled outcome improves on the baseline. This is a finite-sample positive-part plateau; it is not proof of floating-point underflow. No general covariance-calibration or kernel change follows from this audit.

| Acquisition | q | Scores | Fixed-normal MAE | High-sample MAE | Zero fraction |
|---|---|---|---|---|---|
| qei | 1 | 36 | 2.0277e-16 | 2.5363e-05 | 0.83333 |
| qei | 2 | 90 | 4.085e-16 | 4.3587e-05 | 0.73333 |
| qnei | 1 | 36 | 1.5134e-16 | 4.2116e-06 | 0.83333 |
| qnei | 2 | 90 | 3.1363e-16 | 7.8799e-06 | 0.73333 |

Loss aggregation uses Kumulant `MaeLossStat`; zero fractions and expected utilities use Kumulant summaries. BoTorch versions and raw acquisition values are archived alongside the posterior fixture.

## Implementation

- `JointLogExpectedImprovement` and `JointLogNoisyExpectedImprovement` evaluate fat-tailed smoothed improvement directly in log units, using positive-part temperature `1e-6` in original response units and batch-max temperature `.01` in log-utility units. These are smoothed approximations, not simply `log(EI)`. Smoothing keeps all-negative sampled improvements ordered.
- `JointLocalAndRandomSearch` starts with pool coordinate exchange, then moves members using the existing normalized neighborhoods and bounded Klause repair. Every entire candidate set is re-scored, and accepted points retain context, constraints, distinctness, and pending exclusions.
- Joint acquisitions bind completed/pending posterior work, factor its baseline covariance, and draw its latent outcomes once. Candidate sets use a triangular solve and a conditional covariance factor of order at most q. The GP also caches training-to-baseline solves, avoiding repeated baseline-only prediction work. Generic models fall back to full joint prediction.
- Optimized prediction caching is an explicit `JointPosteriorSurrogate` / `LogJointPosteriorSurrogate` capability. Ordinary delegating decorators use their own joint overrides, and do not accidentally inherit an underlying model cache that bypasses their transform.
- BO acquisition, GP conditioning, and feasible-search orchestration stay in Combo. Kumulant retains normal draws, statistical summaries, and Cholesky. Koblas retains matrix products and triangular operations. Neither dependency gains new utilities or GP-specific APIs.

## Baseline-cache workload

A standalone JVM workload compares twenty candidate sets per bound round against full joint prediction/factorization, with identical moments and normals. Binding and baseline factorization are included; fitting is excluded. Two warmup rounds precede three measured repeats, with execution order alternating. This measures BO posterior/sampling orchestration, not a replacement for Kumulant’s factor-kernel benchmark. Times are descriptive measurements on this host, not portable speed guarantees.

| Baseline rows | q | Full round ms | Cached round ms | Ratio | Cached utility MAE |
|---|---|---|---|---|---|
| 16 | 1 | 8.9006 | 9.7759 | 0.91046 | 1.2371e-17 |
| 16 | 4 | 2.4385 | 8.0879 | 0.3015 | 2.0296e-17 |
| 64 | 1 | 11.416 | 2.1313 | 5.3563 | 3.6486e-17 |
| 64 | 4 | 12.388 | 3.0359 | 4.0806 | 4.1366e-16 |
| 128 | 1 | 23.159 | 3.224 | 7.1832 | 0 |
| 128 | 4 | 26.882 | 4.4365 | 6.0594 | 6.245e-18 |

The cached path factors the baseline once, then twenty q-by-q conditional matrices. The full reference factors twenty (baseline+q)-by-(baseline+q) matrices. All compared utilities agree within `1e-8`; raw per-repeat times and scorer errors are archived. The 16-row cases were slower with caching in this run; baseline binding and sampling overhead can outweigh the saved matrix work at small sizes. Larger 64/128-row cases were about 4–7 times faster.

## Whole-loop validation

Validation uses fresh seeds 1701–1706; development seed 951 is excluded and retained separately. The fixed protocol covers a weighted quadratic in 2D, normalized Branin in 2D, a multimodal cosine objective in 3D, and an anisotropic quadratic in 6D. All have a known zero minimum. Noise SD is 0 or .03, with known variances and unit weights. Batch sizes are 1, 2, and 4. All 864 runs exhaust 32 external evaluations; the archive contains 27,648 measurements.

Every method shares the first eight Latin-hypercube points and their measurements. Later exploration probability is .2 (1 for random). Model-driven policies share an independent 16-point pool per seed and round. Pool search uses two starts and one exchange pass; local search adds four sweeps and six attempted neighbors per member, using the same feasible neighborhood machinery for sequential and joint policies. Joint acquisitions use 256 draws. All methods use unchanged default GP fitting and standardized responses. No model or search defaults were selected on validation results.

All methods, including random evaluation, use the same GP posterior-mean recommendation search: 128 independent initial points, four local steps, eight neighbors, and the existing four-start policy. Thus random means random evaluations plus a GP recommendation. Results complete in reverse order inside each batch; no pending work carries between rounds. Dedicated tests cover ongoing pending points, so these runs do not measure asynchronous throughput.

Medians below combine the six seeds, two noise levels, and three batch sizes for each objective (36 outcomes per policy). Smaller regrets are better. Suggestion/recommendation times are reported separately; the random policy fits its GP during recommendation. External budgets are equal, computational budgets are not. Times include JVM/shared-host effects; set evaluations and point evaluations are not equal computational units.

| Objective | Policy | Best evaluated regret | Recommendation regret | Suggest ms | Recommend ms |
|---|---|---|---|---|---|
| quadratic2 | random | 0.0029454 | 0.00025169 | 0.56193 | 3.6784 |
| quadratic2 | sequential-local | 7.8825e-05 | 0.00019782 | 21.384 | 4.6516 |
| quadratic2 | nei-pool | 0.0007849 | 0.00029044 | 15.823 | 3.7012 |
| quadratic2 | lognei-pool | 0.00079445 | 0.00028096 | 26.827 | 3.638 |
| quadratic2 | nei-local | 8.6031e-05 | 2.5233e-05 | 53.761 | 4.1865 |
| quadratic2 | lognei-local | 3.0618e-05 | 0.00027399 | 98.198 | 4.5178 |
| branin2 | random | 0.014736 | 0.01746 | 0.50973 | 3.3057 |
| branin2 | sequential-local | 0.0015715 | 0.0017778 | 18.624 | 3.6066 |
| branin2 | nei-pool | 0.0031835 | 0.0039377 | 15.045 | 3.3791 |
| branin2 | lognei-pool | 0.0031835 | 0.0021677 | 26.43 | 3.3103 |
| branin2 | nei-local | 0.0016856 | 0.0023469 | 49.779 | 3.6733 |
| branin2 | lognei-local | 0.0016811 | 0.0024389 | 91.609 | 3.6718 |
| multimodal3 | random | 0.025013 | 0.027247 | 0.67179 | 4.9246 |
| multimodal3 | sequential-local | 0.018066 | 0.031034 | 28.513 | 5.6556 |
| multimodal3 | nei-pool | 0.016914 | 0.029434 | 16.477 | 4.9304 |
| multimodal3 | lognei-pool | 0.016914 | 0.029051 | 28.532 | 4.6858 |
| multimodal3 | nei-local | 0.014419 | 0.026781 | 59.957 | 5.6428 |
| multimodal3 | lognei-local | 0.014578 | 0.023408 | 101.96 | 5.8147 |
| ellipsoid6 | random | 0.0057365 | 0.0061098 | 1.1679 | 9.4997 |
| ellipsoid6 | sequential-local | 0.0032336 | 0.0037362 | 56.371 | 12.879 |
| ellipsoid6 | nei-pool | 0.0057365 | 0.0075162 | 20.345 | 9.6227 |
| ellipsoid6 | lognei-pool | 0.0057365 | 0.0075162 | 32.287 | 9.5334 |
| ellipsoid6 | nei-local | 0.0048905 | 0.0040268 | 89.588 | 12.607 |
| ellipsoid6 | lognei-local | 0.0042762 | 0.0072795 | 131.59 | 12.009 |

Paired recommendation wins/losses/ties for log-NEI local search; ties use absolute tolerance `1e-10`. Counts are descriptive across correlated settings, not significance tests.

| Objective | vs NEI pool | vs log-NEI pool | vs NEI local | vs sequential local |
|---|---|---|---|---|
| quadratic2 | 24/12/0 | 24/12/0 | 18/17/1 | 15/21/0 |
| branin2 | 21/14/1 | 20/15/1 | 16/16/4 | 18/17/1 |
| multimodal3 | 21/15/0 | 20/16/0 | 19/16/1 | 21/15/0 |
| ellipsoid6 | 22/14/0 | 23/13/0 | 6/13/17 | 16/20/0 |

## Decision and limits

Across all settings, log-NEI local versus NEI pool gives 88/55/1 wins/losses/ties; versus raw NEI local it gives 59/62/23; versus sequential local it gives 70/73/1. Leaving the small pool helps in several settings, but log smoothing does not consistently outperform raw NEI local search. Sequential selection remains competitive, and the six-dimensional problem remains difficult. These results do not justify replacing the default batch policy.

Baseline reuse is a behavior-preserving computational improvement within numerical tolerance. Log acquisition and local joint search remain explicit choices. The finite draw count can still change rankings, and these tests do not integrate GP hyperparameter uncertainty. Baselines are not pruned; one cubic baseline factor and its storage remain per round. Future work should target sampling convergence, higher-dimensional posterior diagnostics, and broader tasks before changing kernels or expanding the surrogate roster.

## Reproduction

Run `./gradlew refineGp` for the suite and `./gradlew profileGpFactorCache` separately for the standalone workload. Use `-PgpRefinement.mode=development -PgpRefinement.seeds=951` for the development smoke run. Run `python3 scripts/summarize_gp_refinement.py` to verify budgets, pairing, initial designs, coordinates, and measured best regret before archiving.

For the independent reference, install the exact versions in `reference.versions.json` (CPU PyTorch) and run `python scripts/audit_gp_reference.py`. Its default input is the archived frozen posterior, preserving that comparison independently of later model fitting. `./gradlew auditGpAcquisition` can regenerate a new posterior export from the old evaluation histories.

`outcomes.csv`, `traces.csv.gz`, `frozen.json.gz`, `reference.csv`, `frozen-scores.csv`, and `factor-cache.csv` contain the complete underlying data. `protocol.json` records fixed settings and both initial/final source hashes. After extracting the explicit cache capability, all 864 non-timing outcome rows and all 27,648 evaluation traces replayed exactly; their hashes are retained. The final full gate `./gradlew check lintDocs --rerun-tasks` passed 268 JVM tests plus API, coverage, lint, and documentation checks. Numerical and behavioral tests compare cached and full predictions, log transforms, decorators, baseline-draw reuse, feasibility/context/exclusion, deterministic and singular covariance, and empty/zero-utility batches.
