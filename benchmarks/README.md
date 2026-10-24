# Bayesian optimization benchmark

Run from the repository root:

```sh
./gradlew jvmTest --tests '*BoBenchmarkTest' --rerun-tasks
python3 scripts/summarize_bo_benchmark.py --output benchmarks/bo-regret.md
```

The [results](bo-regret.md) report median regret after 8, 16, and 32 external evaluations,
final-regret interquartile ranges, and paired seed comparisons. Full per-evaluation
regret-versus-cost traces are written to `build/reports/bo-benchmark/*.csv`.

## Algorithms

- `local-ei`: normalized base coordinates, a full-history Kumulant forest, Gaussian EI
  approximation, and feasible local search after random candidate initialization.
- `pool-ei`: the original normalized pool-only forest/EI control, with local search
  disabled, completed-point repeats allowed, and candidate-order tie-breaking.
- `legacy-pool-ei`: pool-only forest/EI with `LinearFeatureProjection`, reproducing the
  previous candidate-ranking architecture and feature encoding. It uses the current
  study lifecycle, seeds, noise policy, and shared witness sampler; it is not execution
  of an archived Combo binary.
- `random`: the same shared feasible witness generator with no model or acquisition.

All forest variants use eight trees, five initial observations, 10% interleaved random
exploration, 32 initial candidates, and at most 64 initial witness attempts per round.
Local search uses six iterations and twelve attempted neighbors per iteration. Forests
refit against completed history; batching is sequential with pending exclusion and no
fantasies. Trees discover interactions without explicit product features.
The guided optimizer uses its current repeat and tie policies: deterministic completed
points are excluded, noisy replicates remain allowed, and equal scores are ordered
randomly using the round seed. The two pool-only controls retain their original
selection policies so they remain stable references. Separate ablations below isolate
the contribution of the repeat and tie changes.

The seeds are 3, 11, 19, 29, 43, and 61. Each algorithm gets 32 external evaluations per
objective and seed, at unit cost. Acquisition scoring and solver work do not count as
external evaluations. This comparison therefore measures sample efficiency, not wall
clock efficiency. Each algorithm's first seed is repeated to check exact trace replay.

## Objectives

All objectives are maximized and have known optimum zero:

- `quadratic`: `-(x - 0.73)^2`, with `x` in [0, 1].
- `logarithmic`: `-(ln(x / 0.001) / ln(1000))^2`, with `x` in [0.000001, 1] and log geometry.
- `equality`: `-(x - 0.37)^2`, with `x, y` in [0, 1] and the exact constraint `x + y = 1`.
- `mixed`: `-(x - 0.65)^2 - ((count - 4) / 6)^2`, with `x` in [0, 1], integer `count`
  in [1, 6], and a penalty of 0.15 for category `slow` versus `fast`.
- `noisy`: `-(x - 0.31)^2 - 3(y - 0.72)^2`, with both coordinates in [0, 1], plus
  independent uniform observation noise in [-0.05, 0.05]. Noise-aware incumbent handling
  is enabled for every forest variant. The objective noise stream is seeded separately
  and restarted identically for each algorithm.

Regret uses the best noiseless objective value actually evaluated. It does not reward
optimistic predictions or observation-noise outliers. The noisy case measures search
progress, not calibration or the quality of the final model recommendation.

## Limits

These are small synthetic regression benchmarks. Six seeds provide variation but no
statistical significance claim. Tests check feasibility, complete budgets, finite
monotone regret, and reproducibility; they deliberately do not assert an algorithm
ranking. The shared witness generator is a heuristic and is not a uniform sampler of
a constrained region. In particular, independent random proposals can miss an affine
equality's interior; local joint interpolation is intended to cover that case. Forest
variance remains ensemble disagreement rather than a calibrated Bayesian posterior.

## Quadratic EI investigation

```sh
./gradlew jvmTest --tests '*ForestEiDiagnosticsTest' --rerun-tasks
python3 scripts/summarize_forest_ei.py --output benchmarks/forest-ei-investigation.md
```

The [investigation](forest-ei-investigation.md) traces configurations, repeated
evaluations, model versus random rounds, and acquisition plateaus. It compares the
original selection policy, completed-point exclusion alone, random ties alone, both
fixes, and random search. Discovery uses the original six seeds and target 0.73.
Validation uses six different seeds and targets 0.17, 0.43, and 0.86. Full traces are
written to `build/reports/bo-diagnostics/quadratic.csv`; tracing does not add calls to
the acquisition or consume additional search randomness.
