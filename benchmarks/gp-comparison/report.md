# Continuous GP comparison

The batch forest extraction preserves the existing 3,840-row quadratic diagnostic trace byte-for-byte. These additional experiments compare GP/EI, forest/EI, and random search without changing the forest defaults. Frozen-history seeds are 401–408; whole-loop seeds are 301–308. Development seed 91 and preliminary seeds 101–108/201–208 are excluded. No model settings were selected from these final validation results.

## Findings

Across the 48 paired deterministic cases, GP improves best evaluated regret in 48 cases against forest and 48 against random search. Recommendation regret improves in 48 and 48 cases respectively. The smooth quadratic and log-coordinate objectives show especially large model-error reductions. This supports a GP option for these small continuous problems, without changing the forest default.

Noise remains a limitation: at 64 calls, median recommendation regret is 0.00466 for GP and 0.00332 for forest. Paired recommendations remain mixed, as shown below. Better discovery does not guarantee a better noisy recommendation. Noise and recommendation calibration remain a refinement target.

The fixed-history search tables also show nonzero GP acquisition gaps even when prediction error is small. Larger candidate pools sometimes help, but these results do not select a new default search budget.

## Protocol

- Four continuous objectives have true minimum 1: a shifted quadratic, a multimodal squared-sine objective, a noisy anisotropic two-dimensional quadratic, and a quadratic in normalized log coordinates over a variable bounded by 1e-6 and 1.
- Both models see the same frozen histories (16 or 64 points) and 4,096 held-out points. Responses are standardized from training data only. Known noise variance is 0 for deterministic objectives and 0.05²/3 for the noisy objective’s uniform synthetic noise. Truth is used for diagnostics only.
- GP uses the default Matérn-five-halves model, learned per-coordinate length scales and signal variance, and 24 bounded marginal-likelihood search candidates. Row noise is supplied in these benchmarks; inference of unknown noise is covered by unit tests. Its uncertainty conditions on fitted parameters. Forest uses ten bootstrapped randomized all-feature trees and across-tree disagreement. These are model assumptions, not coverage guarantees.
- Whole-loop budgets are 32 or 64 external evaluations. All three algorithms use the same seeded Latin-hypercube start (ten points per dimension capped at 25% of budget). Afterward, random search uses independent uniform draws in encoded geometry; model-based methods interleave the same independent exploration with probability 0.2. Each generated point is checked by Klause. EI search uses 64 initial candidates, four starts, four local steps, and eight neighbors for both models.
- Both models use Gaussian EI on standardized raw responses through the shared acquisition loop. This compares forest/EI with GP/EI; it is not an external SMAC facade comparison. Best regret uses the noiseless reference of evaluated points. Recommendation regret uses model-mean search for GP/forest and best observed response for random search; recommendation truth is diagnostic and is not fed back or charged as an additional evaluation.
- Values are median [25th percentile, 75th percentile] over eight seeds. These small synthetic experiments do not establish a universal ranking or wall-clock performance.

## Frozen model quality

Top-region RMSE uses the best 10% of held-out true objective values for reporting only. Coverage is the fraction of latent truth within two predicted standard deviations.

| Problem | History | Model | RMSE | Top-region RMSE | Two-SD coverage | Mean SD |
| --- | ---: | --- | --- | --- | --- | --- |
| logarithmic | 16 | forest | 0.0216 [0.0187, 0.0242] | 0.00831 [0.00395, 0.0141] | 0.978 [0.932, 0.993] | 0.0291 [0.0243, 0.0325] |
| logarithmic | 16 | gp | 0.000665 [0.000236, 0.000789] | 5.6e-06 [2.53e-06, 3.57e-05] | 1 [1, 1] | 0.000336 [0.000233, 0.000558] |
| logarithmic | 64 | forest | 0.0056 [0.00427, 0.00706] | 0.000697 [0.000446, 0.00095] | 0.97 [0.961, 0.978] | 0.00887 [0.00808, 0.00977] |
| logarithmic | 64 | gp | 1.31e-05 [3.29e-06, 3.83e-05] | 4.45e-11 [1.26e-11, 8.36e-11] | 1 [0.991, 1] | 7.91e-06 [6.37e-06, 9.03e-06] |
| multimodal | 16 | forest | 0.0358 [0.035, 0.0371] | 0.0538 [0.0474, 0.0604] | 0.752 [0.678, 0.786] | 0.0229 [0.0216, 0.025] |
| multimodal | 16 | gp | 0.0287 [0.0257, 0.0346] | 0.0429 [0.0343, 0.0554] | 0.97 [0.786, 1] | 0.0189 [0.0179, 0.0192] |
| multimodal | 64 | forest | 0.014 [0.0129, 0.0156] | 0.012 [0.0101, 0.0164] | 0.937 [0.922, 0.95] | 0.0177 [0.0174, 0.0182] |
| multimodal | 64 | gp | 0.00121 [0.000396, 0.00174] | 2.17e-05 [1.5e-05, 0.000102] | 1 [1, 1] | 0.000889 [0.000654, 0.000941] |
| noisy | 16 | forest | 0.17 [0.15, 0.185] | 0.0937 [0.0786, 0.137] | 0.963 [0.944, 0.978] | 0.202 [0.187, 0.239] |
| noisy | 16 | gp | 0.0595 [0.0525, 0.0641] | 0.0229 [0.0174, 0.0357] | 0.984 [0.975, 0.996] | 0.0513 [0.0474, 0.056] |
| noisy | 64 | forest | 0.0855 [0.084, 0.0882] | 0.0393 [0.0301, 0.0479] | 0.954 [0.938, 0.962] | 0.11 [0.0984, 0.113] |
| noisy | 64 | gp | 0.0228 [0.0196, 0.0247] | 0.0127 [0.0108, 0.0159] | 0.983 [0.956, 0.994] | 0.0199 [0.0188, 0.0204] |
| quadratic | 16 | forest | 0.0246 [0.0217, 0.0259] | 0.00442 [0.00292, 0.00592] | 0.981 [0.952, 0.994] | 0.0339 [0.0278, 0.0371] |
| quadratic | 16 | gp | 0.000448 [0.000271, 0.00068] | 5.58e-06 [3.02e-06, 7.39e-05] | 1 [0.999, 1] | 0.000336 [0.000251, 0.000436] |
| quadratic | 64 | forest | 0.00646 [0.00503, 0.00814] | 0.000596 [0.000407, 0.000791] | 0.969 [0.955, 0.976] | 0.0105 [0.00912, 0.0115] |
| quadratic | 64 | gp | 1.1e-05 [2.09e-06, 2.68e-05] | 3.71e-11 [1.76e-11, 7.9e-11] | 1 [0.997, 1] | 6.76e-06 [5.84e-06, 7.85e-06] |

## Acquisition search on fixed histories

The reference is the largest acquisition value over the fixed 4,096-point pool, not a global maximum. The initial pools of 64 and 256 are nested prefixes. Local search may exceed the finite reference; gaps below zero are clipped for this diagnostic. A larger acquisition value is not necessarily better true objective value.

| Problem | History | Model | Initial pool | Median acquisition gap | Median selected true regret |
| --- | ---: | --- | ---: | ---: | ---: |
| logarithmic | 16 | forest | 64 | 0 | 0.008 |
| logarithmic | 16 | forest | 256 | 0 | 0.00791 |
| logarithmic | 16 | gp | 64 | 1.46e-07 | 1.42e-06 |
| logarithmic | 16 | gp | 256 | 1.3e-07 | 2.38e-06 |
| logarithmic | 64 | forest | 64 | 0 | 0.00246 |
| logarithmic | 64 | forest | 256 | 0 | 0.00264 |
| logarithmic | 64 | gp | 64 | 5.27e-07 | 1.68e-06 |
| logarithmic | 64 | gp | 256 | 4.3e-08 | 6.27e-08 |
| multimodal | 16 | forest | 64 | 0 | 0.0122 |
| multimodal | 16 | forest | 256 | 0 | 0.00859 |
| multimodal | 16 | gp | 64 | 2.57e-06 | 0.0307 |
| multimodal | 16 | gp | 256 | 7.67e-07 | 0.0317 |
| multimodal | 64 | forest | 64 | 0.000297 | 0.00848 |
| multimodal | 64 | forest | 256 | 0 | 0.012 |
| multimodal | 64 | gp | 64 | 5.16e-06 | 0.000127 |
| multimodal | 64 | gp | 256 | 5.18e-06 | 1.91e-05 |
| noisy | 16 | forest | 64 | 0 | 0.0355 |
| noisy | 16 | forest | 256 | 0 | 0.0309 |
| noisy | 16 | gp | 64 | 2.14e-05 | 0.0113 |
| noisy | 16 | gp | 256 | 4.93e-05 | 0.00999 |
| noisy | 64 | forest | 64 | 0.000868 | 0.039 |
| noisy | 64 | forest | 256 | 0 | 0.0397 |
| noisy | 64 | gp | 64 | 5.3e-06 | 0.00196 |
| noisy | 64 | gp | 256 | 0 | 0.00163 |
| quadratic | 16 | forest | 64 | 0 | 0.00352 |
| quadratic | 16 | forest | 256 | 0 | 0.00385 |
| quadratic | 16 | gp | 64 | 1.01e-07 | 2.37e-07 |
| quadratic | 16 | gp | 256 | 6.76e-08 | 2.44e-07 |
| quadratic | 64 | forest | 64 | 0 | 0.0019 |
| quadratic | 64 | forest | 256 | 0 | 0.00181 |
| quadratic | 64 | gp | 64 | 9.24e-07 | 1.86e-06 |
| quadratic | 64 | gp | 256 | 8.34e-07 | 1.51e-06 |

## Equal external budgets

| Problem | Budget | Algorithm | Best evaluated regret | Recommendation regret |
| --- | ---: | --- | --- | --- |
| logarithmic | 32 | forest | 1.28e-06 [5.55e-07, 5.73e-05] | 1.84e-05 [1.06e-06, 6.12e-05] |
| logarithmic | 32 | gp | 1.52e-10 [2.86e-11, 4.26e-10] | 1.93e-10 [2.73e-11, 4.67e-10] |
| logarithmic | 32 | random | 2.93e-05 [1.05e-05, 8.09e-05] | 2.93e-05 [1.05e-05, 8.09e-05] |
| logarithmic | 64 | forest | 9.27e-07 [2.2e-07, 9.57e-06] | 5.33e-07 [5.43e-08, 3.15e-06] |
| logarithmic | 64 | gp | 9.76e-12 [5.8e-12, 4.38e-11] | 6.06e-12 [8.19e-14, 2.54e-11] |
| logarithmic | 64 | random | 4.04e-05 [2.38e-05, 8.11e-05] | 4.04e-05 [2.38e-05, 8.11e-05] |
| multimodal | 32 | forest | 0.00385 [0.0005, 0.00556] | 0.00327 [0.0005, 0.00567] |
| multimodal | 32 | gp | 5.92e-07 [1.78e-07, 1.73e-06] | 1.8e-07 [5.72e-08, 9.37e-07] |
| multimodal | 32 | random | 0.0024 [0.0013, 0.00553] | 0.0024 [0.0013, 0.00553] |
| multimodal | 64 | forest | 0.000757 [7.92e-06, 0.00552] | 0.000921 [4.44e-05, 0.00554] |
| multimodal | 64 | gp | 1.36e-11 [8.7e-12, 5.45e-11] | 2.33e-12 [6.12e-13, 8.12e-12] |
| multimodal | 64 | random | 0.00102 [0.000183, 0.00314] | 0.00102 [0.000183, 0.00314] |
| noisy | 32 | forest | 0.0119 [0.00855, 0.0271] | 0.0274 [0.0176, 0.0403] |
| noisy | 32 | gp | 0.0005 [0.000256, 0.00123] | 0.00196 [0.00104, 0.00436] |
| noisy | 32 | random | 0.0106 [0.0088, 0.0132] | 0.0188 [0.00986, 0.0287] |
| noisy | 64 | forest | 0.00179 [0.00098, 0.005] | 0.00332 [0.00217, 0.0159] |
| noisy | 64 | gp | 0.000583 [0.000167, 0.00214] | 0.00466 [0.0013, 0.00564] |
| noisy | 64 | random | 0.00269 [0.00205, 0.00937] | 0.0172 [0.00798, 0.0265] |
| quadratic | 32 | forest | 5.34e-05 [1.2e-05, 0.000116] | 2.67e-05 [1.45e-05, 0.000107] |
| quadratic | 32 | gp | 9.9e-11 [4.05e-11, 4.98e-10] | 1.41e-10 [1.36e-11, 2.45e-10] |
| quadratic | 32 | random | 4.17e-05 [9.18e-06, 9.58e-05] | 4.17e-05 [9.18e-06, 9.58e-05] |
| quadratic | 64 | forest | 3.32e-06 [1.56e-07, 9.89e-06] | 3.57e-06 [6.76e-07, 9.98e-06] |
| quadratic | 64 | gp | 1.47e-12 [9.09e-14, 1.09e-11] | 6e-12 [8.82e-13, 1.8e-11] |
| quadratic | 64 | random | 6.16e-05 [1.66e-05, 9.26e-05] | 6.16e-05 [1.66e-05, 9.26e-05] |

## Paired comparisons

GP wins/ties/losses count lower regret on the same seed. Differences within 1e-12 are ties.

| Problem | Budget | Comparator | Best evaluated GP W/T/L | Recommendation GP W/T/L |
| --- | ---: | --- | --- | --- |
| logarithmic | 32 | forest | 8/0/0 | 8/0/0 |
| logarithmic | 32 | random | 8/0/0 | 8/0/0 |
| logarithmic | 64 | forest | 8/0/0 | 8/0/0 |
| logarithmic | 64 | random | 8/0/0 | 8/0/0 |
| multimodal | 32 | forest | 8/0/0 | 8/0/0 |
| multimodal | 32 | random | 8/0/0 | 8/0/0 |
| multimodal | 64 | forest | 8/0/0 | 8/0/0 |
| multimodal | 64 | random | 8/0/0 | 8/0/0 |
| noisy | 32 | forest | 8/0/0 | 8/0/0 |
| noisy | 32 | random | 8/0/0 | 8/0/0 |
| noisy | 64 | forest | 6/1/1 | 5/0/3 |
| noisy | 64 | random | 7/1/0 | 7/0/1 |
| quadratic | 32 | forest | 8/0/0 | 8/0/0 |
| quadratic | 32 | random | 8/0/0 | 8/0/0 |
| quadratic | 64 | forest | 8/0/0 | 8/0/0 |
| quadratic | 64 | random | 8/0/0 | 8/0/0 |

## Published-artifact replay

The canonical archive uses the published artifacts listed in [replay.json](replay.json). 193/195 CSV files match the full sibling-source replay byte-for-byte. All 128 model-metric rows and 192 final outcome rows match exactly. One acquisition score differs by 4.75e-13; four late, nearly converged evaluation coordinates differ by at most 1.19e-06, with objective differences at most 4.08e-11. Best-regret traces and final recommendations are unchanged. Either single-library source substitution reproduces the selected frozen-case score; the small numerical difference appears when both siblings are built together. Bitwise identity across dependency compilations is not assumed. The archived replay record retains the individual changed rows.


## Scope and next refinements

Keep GP/EI as an explicit alternative for small continuous histories; retain the forest defaults for existing users. Read the frozen model and fixed-history acquisition tables together before attributing an optimization difference to search or uncertainty. Mixed-type/conditional kernels, censored outcomes, multi-fidelity scheduling, and joint batch EI are unvalidated or unimplemented here. Exact GP fits retain history and scale cubically; coherent lazy function draws also retain their distinct queried points.

The GP follows [GPML chapter 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf). These archived runs used the Koblas factorization API. The current implementation reuses Kumulant’s Cholesky factorization over Koblas BLAS operations; GP models and optimization policy live in Combo.

## Reproduce

```sh
./gradlew compareGp
python3 scripts/summarize_gp_comparison.py
```

Use `-Pgp.mode=frozen` or `-Pgp.mode=optimize` to run one stage. `-Pgp.seeds=91` is a development smoke run and is excluded from this report. Pinned build dependencies define the numerical runtime. Raw diagnostic rows remain under build/reports/gp-comparison; the archived CSVs and compressed traces retain all final validation metrics and external evaluations.
