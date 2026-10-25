# Joint batch acquisition

A paired q=2 experiment compares joint acquisition with existing batch policies. Validation seeds 1501–1508 were reserved before the development run (seed 941); the development results are archived separately and excluded below. No defaults were tuned on these results. Joint policies remain opt-in.

## Protocol

- Objective: `1 + (x - .31)^2 + 3(y - .72)^2` on the unit square, minimized. Noise standard deviation is 0 or .03, with known per-observation variances and unit weights. Kumulant supplies reproducible normal measurement noise.
- Every run has 32 external measurements in 16 batches of two. The initial eight Latin-hypercube points and their measurements match exactly across methods. Subsequent exploration probability is .2 (100% for random sampling). Completion arrives in reverse order within each batch; no pending evaluations remain between rounds.
- Every model uses the unchanged default GP and standardized responses. All five sampling methods use the same posterior-mean GP recommendation search with 64 initial candidates and feasible local search. Thus the random row means random evaluations plus a GP recommendation, not best noisy observed response.
- Model-driven policies share a 16-point independent random candidate pool per seed and round, unaffected by acquisition random consumption. Sequential exclusion and posterior-mean fantasies select from that pool with local search disabled. Joint search uses two starts and one coordinate-exchange pass with 256 antithetic Monte Carlo samples. Assignments are distinct and completed points are excluded. Study validates every proposed assignment through Klause.
- q-EI uses the observed best response when deterministic and the best posterior mean at evaluated coordinates when noisy. q-NEI integrates the latent baseline jointly with candidates. Neither uses true objective values in fitting, search, or recommendation.
- Recommendation regret and best evaluated latent regret measure different decisions. The search is finite-pool and the objective is one smooth two-dimensional example. Eight seeds do not establish broad optimizer rankings. No wall-time speedup is claimed: a whole-set acquisition evaluation is more expensive than a single-point score.

## Validation results

Medians across eight paired seeds; smaller regrets are better. Fits include the final recommendation. Mean-fantasy fits include existing temporary model fits. Acquisition counts measure points for sequential policies and complete sets for joint policies.

| Noise SD | Policy | Best evaluated regret | Recommendation regret | Fits | Acquisition evaluations |
|---|---|---|---|---|---|
| 0.0 | random | 0.0066434 | 0.00010738 | 1 | 0 |
| 0.0 | exclusion | 0.002387 | 9.9098e-05 | 11.5 | 719.5 |
| 0.0 | mean-fantasy | 0.002387 | 0.00011209 | 17 | 719.5 |
| 0.0 | qei | 0.002387 | 0.00011209 | 11.5 | 476.5 |
| 0.0 | qnei | 0.002387 | 0.00010958 | 11.5 | 476.5 |
| 0.03 | random | 0.0066434 | 0.0016752 | 1 | 0 |
| 0.03 | exclusion | 0.002387 | 0.0016051 | 11.5 | 719.5 |
| 0.03 | mean-fantasy | 0.002387 | 0.0022798 | 17 | 719.5 |
| 0.03 | qei | 0.002387 | 0.0032595 | 11.5 | 434.5 |
| 0.03 | qnei | 0.002387 | 0.00447 | 11.5 | 427.5 |

Paired recommendation wins/losses/ties against sequential exclusion (strict numerical comparison):

| Noise SD | Policy | Wins | Losses | Ties | Median paired regret difference |
|---|---|---|---|---|---|
| 0.0 | mean-fantasy | 0 | 2 | 6 | 0 |
| 0.0 | qei | 0 | 3 | 5 | 0 |
| 0.0 | qnei | 1 | 4 | 3 | 3.2278e-07 |
| 0.03 | mean-fantasy | 2 | 3 | 3 | 0 |
| 0.03 | qei | 1 | 6 | 1 | 0.00087294 |
| 0.03 | qnei | 2 | 5 | 1 | 0.0019269 |

Joint acquisition did not improve recommendation regret over sequential exclusion here. Under noise, q-EI won 1/8 paired comparisons and lost 6; q-NEI won 2 and lost 5. The finite pool often gives all model-driven policies the same best evaluated point, while their other measurements change the posterior recommendation. These results support keeping the policies opt-in, not changing the default batch strategy.

## Implementation and limits

Analytic-limit tests verify q=1 normal EI, two-point positive and negative correlation, uncertain-baseline EI, minimization, log-response sampling, and zero covariance. Integration tests cover pending exclusion, context-scoped baselines, frozen-model reuse, state transfer, out-of-order completion, feasibility, partial exhaustion, and atomic reservation failures.

Joint acquisition reuses Kumulant normal sampling, means, and Cholesky, plus Koblas matrix products. Numerical diagonal stabilization is bounded and distinct from observation noise. Exactly duplicate coordinates share one draw and deterministic coordinates receive no artificial noise.

The initial implementation factors each queried joint covariance. No baseline pruning or cached root is implemented, so q-NEI scales poorly to large histories. Finite Monte Carlo EI can have zero-improvement plateaus; log-improvement methods and larger benchmark suites are useful follow-ups. Pending behavior is additionally tested directly; these synchronous runs do not measure asynchronous throughput.

## Reproduction and data

Run `./gradlew compareJointBatch` and `python3 scripts/summarize_joint_batch.py`. Use `-Pjoint.seeds=941` only for the separate development run. `outcomes.csv` contains all 80 validation outcomes, and `traces.csv.gz` contains all 2,560 evaluations with measurement, latent truth, coordinates, and batch. `protocol.json` records source hashes and the base commit. Development outcomes and traces remain under `development/`.
