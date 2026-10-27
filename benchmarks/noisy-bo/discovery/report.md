These are protocol-discovery runs with a single-evaluation initial-design allowance for racing. The final study uses the existing measurement-aware design and separate seeds.

# Noisy BO refinement

Validation uses fresh seeds 1001–1012; development seeds 911–914 are excluded. No GP/forest/search defaults were selected from these validation results. The lifecycle fix makes model inspection independent of future suggestion randomness.

## Protocol

- Objective: `1 + (x - .31)^2 + 3(y - .72)^2`, with independent Gaussian measurement noise of standard deviation .03 or .1. Every observation supplies its known noise variance. Noise comes from Kumulant’s normal sampler with a reproducible configuration/measurement-key seed.
- Budgets are 32/64 external measurements including all racing and confirmation calls. All 240 runs exhaust their budget; the archive contains 11,520 raw evaluations and coordinates.
- GP uses the unchanged default Matérn model and 24 likelihood candidates. Forest uses ten bootstrapped trees, random thresholds, both features, and ensemble disagreement. Responses are standardized; the objective is minimized. All proposals obey Klause feasibility.
- Ordinary GP/forest runs use the same budget-aware Latin-hypercube design and .2 independent exploration. Acquisition search has 64 initial candidates, four starts, four steps, eight neighbors. Recommendation compares that search with 1,024 initial candidates on the same cached model.
- Racing uses four search keys, a one-measurement initial incumbent/challenger, and a two-standard-error rejection allowance. Confirmation reserves nine calls: three finalists on three fresh keys. Observed/model-confirm share identical search histories; only the frozen shortlist ranking changes. The final winner uses only fresh responses. Plain race spends all calls on search/racing.
- Diagnostics use 4,096 fresh query points plus history and returned recommendations for a finite posterior-mean reference. The reference includes the returned candidates and is not a global optimum. Truth is used only for evaluation/top-decile diagnostics, never fitting or finalist ranking. Observation coverage includes known measurement variance; latent coverage excludes it. Forest coverage uses a Gaussian approximation to ensemble dispersion.
- Prediction losses use Kumulant MseLossStat; means/coverage use MeanStat. Recommendation search consumes additional computation, with no external observations. No wall-clock or universal performance claims are made.

## Recommendation and evaluated-point regret

Values are medians over twelve paired seeds; lower is better.

| Calls | Noise SD | Policy | Best evaluated regret | Recommendation regret | Larger-search regret |
| ---: | ---: | --- | ---: | ---: | ---: |
| 32 | 0.03 | forest | 0.0080466 | 0.01611 | 0.014371 |
| 32 | 0.03 | gp | 0.00027457 | 0.00079283 | 0.0010552 |
| 32 | 0.03 | model-confirm | 0.04116 | 0.04116 | 0.04116 |
| 32 | 0.03 | observed-confirm | 0.04116 | 0.04116 | 0.04116 |
| 32 | 0.03 | race | 0.0042935 | 0.0055759 | 0.0055759 |
| 32 | 0.1 | forest | 0.0092925 | 0.047426 | 0.047958 |
| 32 | 0.1 | gp | 0.0027739 | 0.010188 | 0.010546 |
| 32 | 0.1 | model-confirm | 0.032275 | 0.048103 | 0.048103 |
| 32 | 0.1 | observed-confirm | 0.032275 | 0.048103 | 0.048103 |
| 32 | 0.1 | race | 0.013411 | 0.041253 | 0.041253 |
| 64 | 0.03 | forest | 0.001353 | 0.011818 | 0.011686 |
| 64 | 0.03 | gp | 0.00029693 | 0.0019929 | 0.0024226 |
| 64 | 0.03 | model-confirm | 0.00069422 | 0.0011122 | 0.0011122 |
| 64 | 0.03 | observed-confirm | 0.00069422 | 0.0011818 | 0.0011818 |
| 64 | 0.03 | race | 0.00057128 | 0.0012799 | 0.0012799 |
| 64 | 0.1 | forest | 0.0039847 | 0.041906 | 0.042008 |
| 64 | 0.1 | gp | 0.000449 | 0.006468 | 0.006567 |
| 64 | 0.1 | model-confirm | 0.0021151 | 0.0029669 | 0.0029669 |
| 64 | 0.1 | observed-confirm | 0.0021151 | 0.0024064 | 0.0024064 |
| 64 | 0.1 | race | 0.0018263 | 0.0024064 | 0.0024064 |

For racing policies the last two columns both report the empirical incumbent; there is no larger posterior-mean search in those policies.

## Model and recommendation-search diagnostics

| History | Noise SD | Model | RMSE | Top RMSE | Latent coverage | Top latent coverage | Observation coverage | Search gap | Larger-search gap |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.03 | forest | 0.17128 | 0.040751 | 0.92908 | 0.98171 | 0.94214 | 0 | 0 |
| 32 | 0.03 | gp | 0.081255 | 0.015485 | 0.94678 | 1 | 0.93909 | 0 | 1.5675e-05 |
| 32 | 0.1 | forest | 0.17135 | 0.096509 | 0.92834 | 0.96463 | 0.95007 | 0 | 0 |
| 32 | 0.1 | gp | 0.12477 | 0.033496 | 0.94287 | 1 | 0.93872 | 0 | 6.8174e-06 |
| 64 | 0.03 | forest | 0.1118 | 0.029831 | 0.95117 | 0.98171 | 0.96155 | 0 | 0 |
| 64 | 0.03 | gp | 0.040293 | 0.012097 | 0.96167 | 1 | 0.95508 | 0 | 1.4893e-05 |
| 64 | 0.1 | forest | 0.13565 | 0.059933 | 0.95007 | 0.97195 | 0.96814 | 0 | 0 |
| 64 | 0.1 | gp | 0.071637 | 0.032992 | 0.97449 | 1 | 0.95984 | 6.3167e-06 | 0 |

## Paired recommendation comparisons

Counts are wins/ties/losses with a 1e-12 tie tolerance.

| Calls | Noise SD | Comparison | Wins/ties/losses |
| ---: | ---: | --- | --- |
| 32 | 0.03 | gp vs forest | 11/0/1 |
| 32 | 0.03 | model-confirm vs observed-confirm | 0/12/0 |
| 32 | 0.03 | model-confirm vs gp | 1/0/11 |
| 32 | 0.1 | gp vs forest | 11/0/1 |
| 32 | 0.1 | model-confirm vs observed-confirm | 0/12/0 |
| 32 | 0.1 | model-confirm vs gp | 1/0/11 |
| 64 | 0.03 | gp vs forest | 11/0/1 |
| 64 | 0.03 | model-confirm vs observed-confirm | 1/11/0 |
| 64 | 0.03 | model-confirm vs gp | 11/0/1 |
| 64 | 0.1 | gp vs forest | 10/0/2 |
| 64 | 0.1 | model-confirm vs observed-confirm | 1/10/1 |
| 64 | 0.1 | model-confirm vs gp | 8/0/4 |

## Decisions and limits

Keep model-based confirmation and independent recommendation search opt-in. A smaller acquisition-search gap does not establish a better true recommendation. Calibration conditions on fitted hyperparameters and model assumptions; this study does not supply a universal variance correction or integrate hyperparameter uncertainty. The two noise levels share one smooth two-dimensional objective, so validation is narrow.

## Reproduce

```sh
./gradlew refineNoisyBo
python3 scripts/summarize_noisy_bo.py
```

Use `-Pnoisy.seeds=911` for development smoke runs, which cannot generate the validation report.
