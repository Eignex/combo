# BO refinement cycle

Frozen-history diagnostics use 20 seeds (201–220). Whole-loop validation uses 12 different seeds (101–112), with separate runs budgeted at 32 and 128 external calls. Development seed 91 is excluded. Lower regret and prediction error are better. No algorithm ranking is asserted by tests.

## Decisions

- Keep disagreement-only uncertainty as the default. Extra leaf-mean variance widens coverage without changing fitted means and does not consistently improve recommendation regret.
- Keep racing and confirmation opt-in. Accounting for measurements fixes design allocation, but guarded rejection still spends too much of a 32-call budget on repeats. Fresh-key confirmation helps some guarded runs and does not consistently beat ordinary search.
- Do not raise the default candidate pool to 10,000 based on these tests. The quadratic’s median reference-pool acquisition gap is already zero at 64 candidates. Noisy cases have search misses, but acquisition maximization and actual objective improvement remain different measurements.

## Frozen models

Every model receives identical independently sampled raw-coordinate histories and predicts the same 10,000 held-out points. These diagnostics use raw positive responses, not the log-response whole-loop profile. Coverage is the fraction of noiseless reference targets within two predicted standard deviations; it is descriptive coverage, not a confidence guarantee. Top-region metrics use the best 10% of held-out true costs for reporting only. Known noise is supplied to Combo’s leaf-mean estimator, while SMAC and disagreement mode do not use it.

| Problem | History | Model | Median RMSE | Median coverage | Median SD | Top-region RMSE | Top-region coverage |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| noisy | 32 | Disagreement | 0.1029 | 0.9482 | 0.1529 | 0.05295 | 0.972 |
| noisy | 32 | LeafMean | 0.1029 | 0.9698 | 0.1715 | 0.05295 | 1 |
| noisy | 32 | smac-2.4.1 | 0.1148 | 0.8328 | 0.1046 | 0.04878 | 0.8676 |
| noisy | 128 | Disagreement | 0.05525 | 0.9649 | 0.08336 | 0.02848 | 0.979 |
| noisy | 128 | LeafMean | 0.05525 | 0.9896 | 0.09489 | 0.02848 | 1 |
| noisy | 128 | smac-2.4.1 | 0.06827 | 0.8676 | 0.05981 | 0.02759 | 0.9046 |
| quadratic | 32 | Disagreement | 0.01209 | 0.9546 | 0.01911 | 0.002108 | 1 |
| quadratic | 32 | LeafMean | 0.01209 | 0.9751 | 0.02089 | 0.002108 | 1 |
| quadratic | 32 | smac-2.4.1 | 0.01485 | 0.8454 | 0.0122 | 0.001048 | 0.8876 |
| quadratic | 128 | Disagreement | 0.002916 | 0.9825 | 0.005667 | 0.0002136 | 0.9895 |
| quadratic | 128 | LeafMean | 0.002916 | 0.9959 | 0.006225 | 0.0002136 | 1 |
| quadratic | 128 | smac-2.4.1 | 0.003394 | 0.8317 | 0.003353 | 0.0003646 | 0.7063 |

## Acquisition search at fixed histories

Only the initial candidate budget varies; each run retains four adaptive starts, four local steps, and eight attempted neighbors. Initial pools are nested prefixes of the same 10,000 feasible held-out points. The reference maximum is over that finite pool, not a global optimum. Local search can exceed it. The gap is clipped at zero only for this diagnostic.

| Problem | History | Uncertainty | Candidates | Median acquisition gap | Median selected true cost |
| --- | ---: | --- | ---: | ---: | ---: |
| noisy | 32 | Disagreement | 64 | 0.0004249 | 1.125 |
| noisy | 32 | Disagreement | 256 | 0 | 1.134 |
| noisy | 32 | Disagreement | 10000 | 0 | 1.111 |
| noisy | 32 | LeafMean | 64 | 3.661e-05 | 1.102 |
| noisy | 32 | LeafMean | 256 | 0 | 1.117 |
| noisy | 32 | LeafMean | 10000 | 0 | 1.087 |
| noisy | 128 | Disagreement | 64 | 0.0002363 | 1.014 |
| noisy | 128 | Disagreement | 256 | 5.655e-05 | 1.019 |
| noisy | 128 | Disagreement | 10000 | 0 | 1.018 |
| noisy | 128 | LeafMean | 64 | 0.000325 | 1.014 |
| noisy | 128 | LeafMean | 256 | 0.0005051 | 1.015 |
| noisy | 128 | LeafMean | 10000 | 0 | 1.014 |
| quadratic | 32 | Disagreement | 64 | 0 | 1.004 |
| quadratic | 32 | Disagreement | 256 | 0 | 1.004 |
| quadratic | 32 | Disagreement | 10000 | 0 | 1.004 |
| quadratic | 32 | LeafMean | 64 | 0 | 1.004 |
| quadratic | 32 | LeafMean | 256 | 0 | 1.004 |
| quadratic | 32 | LeafMean | 10000 | 0 | 1.004 |
| quadratic | 128 | Disagreement | 64 | 0 | 1.001 |
| quadratic | 128 | Disagreement | 256 | 0 | 1 |
| quadratic | 128 | Disagreement | 10000 | 0 | 1 |
| quadratic | 128 | LeafMean | 64 | 0 | 1.001 |
| quadratic | 128 | LeafMean | 256 | 0 | 1.001 |
| quadratic | 128 | LeafMean | 10000 | 0 | 1.001 |

## Whole-loop validation

All model-driven variants use the same bounded search profile: ten randomized all-feature trees, log-response EI, 64 initial candidates, four starts, four steps, eight neighbors, and 20% independent exploration. This is smaller than the original 256-candidate/ten-start SMAC comparison and does not replace those historical results. The conditional problem combines a real variable, category, and optional integer subspace. Every proposal is checked in source coordinates.

| Problem | Calls | Algorithm | Median discovery regret | Median recommendation regret | Recommendation IQR | Median configurations |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| conditional | 32 | disagreement | 0.01117 | 0.01117 | 0.0001813–0.02087 | 32 |
| conditional | 32 | leaf | 0.003354 | 0.003354 | 0.0003578–0.02 | 32 |
| conditional | 32 | random | 0.02011 | 0.02011 | 0.004381–0.02531 | 29.5 |
| conditional | 128 | disagreement | 8.416e-06 | 8.416e-06 | 3.177e-06–1.415e-05 | 128 |
| conditional | 128 | leaf | 4.922e-06 | 4.922e-06 | 1.596e-06–7.961e-06 | 128 |
| conditional | 128 | random | 0.0037 | 0.0037 | 0.003094–0.004121 | 113 |
| noisy | 32 | budgeted-race | 0.00286 | 0.01594 | 0.006104–0.05157 | 23.5 |
| noisy | 32 | confirmation | 0.005828 | 0.01241 | 0.008187–0.02039 | 26 |
| noisy | 32 | confirmed-race | 0.02442 | 0.02442 | 0.01383–0.07989 | 11 |
| noisy | 32 | disagreement | 0.002945 | 0.00878 | 0.002035–0.02105 | 32 |
| noisy | 32 | guarded-race | 0.02003 | 0.03636 | 0.02413–0.07575 | 14 |
| noisy | 32 | leaf | 0.004553 | 0.01291 | 0.007224–0.03181 | 32 |
| noisy | 32 | legacy-race | 0.01264 | 0.0214 | 0.0191–0.02913 | 24 |
| noisy | 32 | random | 0.02259 | 0.02307 | 0.02259–0.02927 | 29.5 |
| noisy | 128 | budgeted-race | 0.001084 | 0.01104 | 0.005408–0.01683 | 106 |
| noisy | 128 | confirmation | 0.0008595 | 0.00399 | 0.00135–0.007599 | 104 |
| noisy | 128 | confirmed-race | 0.002953 | 0.009372 | 0.005441–0.01117 | 45 |
| noisy | 128 | disagreement | 0.0007752 | 0.003877 | 0.001425–0.00707 | 128 |
| noisy | 128 | guarded-race | 0.001666 | 0.005634 | 0.002032–0.009966 | 55 |
| noisy | 128 | leaf | 0.0007722 | 0.006114 | 0.003061–0.009504 | 128 |
| noisy | 128 | legacy-race | 0.001247 | 0.008713 | 0.001934–0.01225 | 105.5 |
| noisy | 128 | random | 0.005952 | 0.008027 | 0.006175–0.01936 | 115.5 |
| quadratic | 32 | disagreement | 3.469e-05 | 3.469e-05 | 7.157e-06–0.0002347 | 32 |
| quadratic | 32 | leaf | 1.814e-05 | 1.814e-05 | 3.593e-06–0.0002558 | 32 |
| quadratic | 32 | random | 0.0002145 | 0.0002145 | 0.0001053–0.0004122 | 29.5 |
| quadratic | 128 | disagreement | 2.07e-08 | 2.07e-08 | 7.796e-09–6.551e-08 | 128 |
| quadratic | 128 | leaf | 9.733e-08 | 9.733e-08 | 1.646e-08–1.317e-06 | 128 |
| quadratic | 128 | random | 5.351e-05 | 5.351e-05 | 1.555e-05–8.422e-05 | 114 |

## Paired recommendation comparisons

| Problem | Calls | Candidate | Reference | Wins / ties / losses |
| --- | ---: | --- | --- | --- |
| quadratic | 32 | leaf | disagreement | 5 / 2 / 5 |
| quadratic | 128 | leaf | disagreement | 3 / 0 / 9 |
| conditional | 32 | leaf | disagreement | 6 / 0 / 6 |
| conditional | 128 | leaf | disagreement | 6 / 0 / 6 |
| noisy | 32 | leaf | disagreement | 3 / 1 / 8 |
| noisy | 32 | budgeted-race | legacy-race | 6 / 0 / 6 |
| noisy | 32 | guarded-race | budgeted-race | 4 / 0 / 8 |
| noisy | 32 | confirmed-race | guarded-race | 6 / 4 / 2 |
| noisy | 32 | confirmation | disagreement | 3 / 6 / 3 |
| noisy | 128 | leaf | disagreement | 4 / 0 / 8 |
| noisy | 128 | budgeted-race | legacy-race | 5 / 0 / 7 |
| noisy | 128 | guarded-race | budgeted-race | 7 / 0 / 5 |
| noisy | 128 | confirmed-race | guarded-race | 4 / 2 / 6 |
| noisy | 128 | confirmation | disagreement | 5 / 2 / 5 |

## Policies and limits

- `disagreement` retains the existing forest uncertainty. `leaf` adds approximate leaf-mean sampling variance using original observation units. Their frozen fitted means are identical. Leaf-mean uncertainty can overlap tree disagreement and can overcover; it remains opt-in.
- `legacy-race` retains full-plan initial incumbent evaluation and paired-mean rejection. `budgeted-race` only divides the initial configuration allowance by the three planned measurement keys. `guarded-race` additionally starts incumbents with two keys and uses a two-standard-error allowance before early rejection. Repeated error-band comparisons are not sequential significance tests.
- `confirmed-race` adds a frozen top-three shortlist and a fresh-key final comparison. `confirmation` applies that final comparison to ordinary one-measurement search. Each finalist receives two fresh measurements at budget 32, or eight at budget 128; search stops at 26 or 104 calls. The final recommendation uses only fresh-key outcomes. Every confirmation call counts toward the same external budget and trains once.
- Noisy costs are `1 + (x-.31)^2 + 3*(y-.72)^2` plus a fixed seed/configuration pseudo-noise function of amplitude .05; supplied variance is .0025/3. Different measurement keys produce paired reproducible outcomes. This is one synthetic noise process; heavy tails, drifting noise, and varying instance difficulty need further validation. Oracle costs are used for reporting only.
- The [pinned SMAC 2.4.1 sklearn backend](https://github.com/automl/SMAC3/blob/v2.4.1/smac/model/random_forest/random_forest.py) uses variance across tree predictions. The previous report’s claim that it adds within-leaf variance was incorrect and has been corrected. The frozen reference uses SMAC’s HPO tree settings with log_y disabled to match these raw responses; encoding is identical, so differences in its ConfigSpace categorical handling are absent here.
- The first smoke run with an optional real subspace was interrupted after exact LP certification dominated runtime. The final conditional case uses an optional integer subspace. This study does not establish performance on optional-real constrained spaces or compare wall-clock speed.
- [Final outcomes](outcomes.csv), aggregate diagnostics, and [compressed complete traces](traces.csv.gz) are archived beside this report. Frozen numeric datasets remain under `build/reports/bo-refinement/` and are reproducible from the runner and seed lists. See [reproduction instructions](../README.md#refinement-cycle).
