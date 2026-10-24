# Pinned SMAC comparison

SMAC 2.4.1 HPO facade versus the new Combo profile and feasible random search. Six paired seeds; 32 external objective calls per run, including repetitions. All costs are minimized and have noiseless optimum one. Lower regret is better.

## Discovery regret

| Problem | Algorithm | Median @8 | Median @16 | Median @32 | Final IQR |
| --- | --- | ---: | ---: | ---: | --- |
| logarithmic | combo-profile | 0.000459 | 2.46e-05 | 1.07e-05 | 7.71e-06–3.28e-05 |
| logarithmic | random | 0.000662 | 0.00059 | 0.000151 | 7.23e-05–0.000771 |
| logarithmic | smac-2.4.1 | 0.00298 | 0.000405 | 1.09e-05 | 1.16e-07–7.21e-05 |
| mixed | combo-profile | 0.0393 | 0.0193 | 0.000372 | 0.000161–0.00121 |
| mixed | random | 0.0816 | 0.0202 | 0.00887 | 0.00887–0.00887 |
| mixed | smac-2.4.1 | 0.0383 | 0.0259 | 0.0256 | 0.00831–0.0505 |
| noisy | combo-profile | 0.0488 | 0.00444 | 0.00208 | 0.00176–0.00376 |
| noisy | combo-racing | 0.0697 | 0.0488 | 0.0174 | 0.00961–0.0242 |
| noisy | random | 0.0408 | 0.0176 | 0.00618 | 0.00494–0.00756 |
| noisy | smac-2.4.1 | 0.062 | 0.0333 | 0.0185 | 0.00873–0.0251 |
| quadratic | combo-profile | 0.000812 | 0.000149 | 1.14e-05 | 4.69e-06–1.62e-05 |
| quadratic | random | 0.00123 | 0.00107 | 0.00018 | 0.000114–0.000619 |
| quadratic | smac-2.4.1 | 0.00213 | 2.58e-05 | 5.46e-07 | 1.62e-07–2.86e-06 |

## Paired final discovery comparisons

| Problem | Combo vs random wins/ties/losses | Combo vs SMAC wins/ties/losses |
| --- | --- | --- |
| quadratic | 5/0/1 | 2/0/4 |
| logarithmic | 4/0/2 | 2/0/4 |
| mixed | 5/0/1 | 5/0/1 |
| noisy | 5/0/1 | 6/0/0 |

## Noisy recommendations

These use each algorithm’s selected **evaluated** configuration, scored without observation noise for reporting only. Combo without racing and random search select the lowest measured replicate mean; racing uses its fully confirmed incumbent; SMAC uses its intensifier incumbent. This differs from a search over the surrogate mean and may be worse than the best point discovered.

| Algorithm | Median recommendation regret @32 | Final IQR | Median distinct configurations |
| --- | ---: | --- | ---: |
| combo-profile | 0.00951 | 0.00737–0.0105 | 32 |
| combo-racing | 0.0324 | 0.0174–0.0412 | 20 |
| random | 0.0154 | 0.00781–0.019 | 27.5 |
| smac-2.4.1 | 0.027 | 0.0109–0.0414 | 24 |

## Method and limits

- Full per-evaluation coordinates, measured responses, discovery/regret, recommendation regret, measurement seeds, and costs are archived in [smac-traces.csv](smac-traces.csv). The first seed of every Combo case is rerun with an exact non-timing trace assertion. Timing is diagnostic only: runs overlapped builds and do not establish wall-clock rankings.
- See [reproduction instructions](README.md#pinned-smac-comparison) and the locked Python environment. SMAC uses its stock HPO facade: Sobol design (10 per dimension, capped at 25% of trials), 20% random challengers, 10 randomized all-feature trees, log-cost EI, 10 local starts, 10,000 initial challengers, and an intensifier with at most 3 measurements per configuration.
- Combo uses a continuous Latin hypercube with the same dimension/budget sizing, 20% independent feasible exploration, 10 randomized all-feature Kumulant trees, lognormal EI, 256 initial candidates, 10 adaptive local walks, 8 steps per walk, and 12 attempted neighbors per step. Internal computation budgets and forest statistics differ; equal external budgets do not imply implementation parity. Both pinned forest backends use across-tree disagreement. Leaf-response aggregation and fitting still differ.
- Deterministic objectives are quadratic, logarithmic, and mixed (real, integer, category). They match the legacy benchmark geometry, shifted to strictly positive costs. The noisy objective is `1 + (x - .31)^2 + 3*(y - .72)^2` plus a fixed configuration/seed pseudo-noise function with amplitude .05. The three noisy measurement seeds are read from SMAC’s measurement identities; Combo never reads its outcomes. Plain Combo and random search use the first identity for each novel point; racing and SMAC repeat points on matched identities. The oracle is used only for reporting.
- Six seeds and small synthetic objectives do not establish statistical significance or general superiority. Discovery and final recommendation measure different goals. Racing is optional and can spend a short budget confirming points at the expense of discovery. Its early rejection uses paired means with a reproducibly shuffled challenger key order, without confidence bounds. Instance-feature marginalization, censored runtimes, Hyperband/multi-fidelity promotion, and joint batch acquisition remain absent. Exact affine feasibility is checked separately by the constrained regression benchmarks.
