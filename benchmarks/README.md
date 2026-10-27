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
Local search uses four adaptive starts, six iterations per start, and twelve attempted neighbors per iteration. Forests
refit against completed history; batching is sequential with pending exclusion and no
fantasies. Trees discover interactions without explicit product features.
The guided optimizer uses its current repeat and tie policies: deterministic completed
points are excluded, noisy replicates remain allowed, and equal scores are ordered
randomly using the round seed. The two pool-only controls retain their original
selection policies as policy controls. All variants now use independent model-free exploration; these are current-code controls, not archived executions. Separate ablations below isolate
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
a constrained region. Coordinate-wise random proposals can miss an affine equality's interior; checked joint
interpolation in shared sampling and local search supplies additional coverage. Forest
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

## Pinned SMAC comparison

Run the actual SMAC facade in an isolated Python 3.12 environment, then Combo against
its published Kumulant dependency:

```sh
uv venv --python 3.12 build/smac-venv
uv pip install --python build/smac-venv/bin/python -r benchmarks/smac-requirements.txt
PYTHONHASHSEED=0 build/smac-venv/bin/python scripts/run_smac_reference.py
./gradlew compareSmac
python3 scripts/summarize_smac_comparison.py
```

The [comparison](smac-comparison.md) reports discovery regret at 8/16/32 calls,
interquartile ranges, paired outcomes, and final noisy recommendation quality. Raw
traces are archived in [smac-traces.csv](smac-traces.csv); local state/checkpoints live
under `build/reports/smac-comparison/`. The Python hash seed is required because
configuration hashes affect candidate iteration order. Exact timing is not compared.

Both runners use six seeds (3, 11, 19, 29, 43, 61) and a fixed 32-call budget. The
reference uses SMAC 2.4.1's standard HPO facade with its default design, search, and
intensifier settings. Combo tests `smacProfile`, the shared feasible random baseline,
and optional paired racing on the noisy objective. Randomized trees and lognormal EI
come directly from Kumulant; Combo handles lifecycle, coordinates, and search.
The legacy negative-reward benchmarks above remain separate from this positive-cost
log-response comparison. The EI diagnostics explicitly retain one non-adaptive local
start to isolate repeat/tie policies from the new multi-start search.

## Refinement cycle

```sh
./gradlew refineBo -Prefinement.mode=frozen
PYTHONHASHSEED=0 build/smac-venv/bin/python scripts/run_smac_frozen.py
./gradlew scoreSmacFrozen
./gradlew refineBo -Prefinement.mode=optimize
./gradlew refineBo -Prefinement.mode=race
python3 scripts/summarize_bo_refinement.py
```

The [study](refinement/report.md) archives aggregate frozen-model/search diagnostics and
complete validation traces. Frozen histories use 20 seeds, 201–220. Whole-loop runs
use 12 different seeds, 101–112, and budgets of 32 and 128 calls. Development seed 91
is excluded. `-Prefinement.seeds=91` runs a smoke sample. The model-only reference uses
the same pinned Python environment as the original SMAC comparison. Python exports
raw predictions; `scoreSmacFrozen` computes prediction losses with Kumulant scorer
stats and writes `smac-models.csv`.

Frozen models see identical training rows and predict the same 10,000 independent
feasible points. Candidate-budget comparisons keep the model fixed and vary nested
initial pools of 64, 256, and 10,000 points. Whole-loop validation keeps a smaller,
fixed acquisition budget across all variants and includes an optional integer
subspace. The noisy runs separately ablate measurement-aware initial design,
progressive/guarded racing, and a frozen final comparison on unused measurement keys.
No report asserts a stochastic ranking as a test requirement, and the oracle never
supplies labels or suggestions to an optimizer.

## Continuous Gaussian-process comparison

Run `./gradlew compareGp` and `python3 scripts/summarize_gp_comparison.py` for
frozen-history prediction/search diagnostics and equal-budget GP/EI, forest/EI, and
random search. The [report](gp-comparison/report.md) archives fresh validation seeds,
paired outcomes, and evaluation traces. `-Pgp.mode=frozen` or `-Pgp.mode=optimize`
selects one stage; `-Pgp.seeds=91` is an excluded development smoke run.

## Noisy recommendations

```sh
./gradlew refineNoisyBo
./gradlew auditLegacyNoisyBo
python3 scripts/summarize_noisy_bo.py
```

The [noisy refinement report](noisy-bo/report.md) separates model error, latent versus
observation coverage, finite-pool recommendation-search gaps, and fresh confirmation.
Validation uses seeds 1301–1312, budgets 32/64, and Gaussian noise SD .03/.1.
Development seeds 911–914 are excluded. The legacy audit refits the archived 64-call
histories separately; it does not replay the earlier solver population or tune defaults.

Protocol-discovery seeds 1001–1012 are archived separately with the original single-evaluation
initial-design allowance; reproduce that protocol using `-Pnoisy.design=legacy` and explicit seeds.
