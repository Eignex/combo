# Noisy BO refinement

Validation uses fresh seeds 1301–1312; development seeds 911–914 are excluded. No GP/forest/search defaults were selected from these validation results. The lifecycle fix makes model inspection independent of future suggestion randomness.

## Protocol

- Objective: `1 + (x - .31)^2 + 3(y - .72)^2`, with independent Gaussian measurement noise of standard deviation .03 or .1. Every observation supplies its known noise variance. Noise comes from Kumulant’s normal sampler with a reproducible configuration/measurement-key seed.
- Budgets are 32/64 external measurements including all racing and confirmation calls. All 240 runs exhaust their budget; the archive contains 11,520 raw evaluations and coordinates.
- GP uses the unchanged default Matérn model and 24 likelihood candidates. Forest uses ten bootstrapped trees, random thresholds, both features, and ensemble disagreement. Responses are standardized; the objective is minimized. All proposals obey Klause feasibility.
- Ordinary GP/forest runs use the same budget-aware Latin-hypercube design and .2 independent exploration. Acquisition search has 64 initial candidates, four starts, four steps, eight neighbors. Recommendation compares that search with 1,024 initial candidates on the same cached model.
- Racing initial designs use the existing three-measurements-per-configuration allowance, capped against the search budget after reserving confirmation calls. Racing uses four search keys, a one-measurement initial incumbent/challenger, and a two-standard-error rejection allowance. Confirmation reserves nine calls: three finalists on three fresh keys. Observed/model-confirm share identical search histories; only the frozen shortlist ranking changes. The final winner uses only fresh responses. Plain race spends all calls on search/racing.
- Diagnostics use 4,096 fresh query points plus history and returned recommendations for a finite posterior-mean reference. The reference includes the returned candidates and is not a global optimum. Truth is used only for evaluation/top-decile diagnostics, never fitting or finalist ranking. Observation coverage includes known measurement variance; latent coverage excludes it. Forest coverage uses a Gaussian approximation to ensemble dispersion.
- Prediction losses use Kumulant MseLossStat; means/coverage use MeanStat. Recommendation search consumes additional computation, with no external observations. No wall-clock or universal performance claims are made.

## Recommendation and evaluated-point regret

Values are medians over twelve paired seeds; lower is better.

| Calls | Noise SD | Policy | Best evaluated regret | Recommendation regret | Larger-search regret |
| ---: | ---: | --- | ---: | ---: | ---: |
| 32 | 0.03 | forest | 0.0060617 | 0.012933 | 0.016612 |
| 32 | 0.03 | gp | 0.00036326 | 0.0025255 | 0.0025008 |
| 32 | 0.03 | model-confirm | 0.0098916 | 0.010103 | 0.010103 |
| 32 | 0.03 | observed-confirm | 0.0098916 | 0.010103 | 0.010103 |
| 32 | 0.03 | race | 0.0055293 | 0.0068607 | 0.0068607 |
| 32 | 0.1 | forest | 0.0083612 | 0.035561 | 0.031724 |
| 32 | 0.1 | gp | 0.002275 | 0.0059458 | 0.0047489 |
| 32 | 0.1 | model-confirm | 0.077421 | 0.096977 | 0.096977 |
| 32 | 0.1 | observed-confirm | 0.077421 | 0.096977 | 0.096977 |
| 32 | 0.1 | race | 0.0096631 | 0.023691 | 0.023691 |
| 64 | 0.03 | forest | 0.0016104 | 0.011988 | 0.009621 |
| 64 | 0.03 | gp | 0.00029558 | 0.0019993 | 0.0021294 |
| 64 | 0.03 | model-confirm | 0.00070004 | 0.0025532 | 0.0025532 |
| 64 | 0.03 | observed-confirm | 0.00070004 | 0.0022981 | 0.0022981 |
| 64 | 0.03 | race | 0.00021989 | 0.0021146 | 0.0021146 |
| 64 | 0.1 | forest | 0.0073232 | 0.036355 | 0.036355 |
| 64 | 0.1 | gp | 0.0011681 | 0.0061526 | 0.005606 |
| 64 | 0.1 | model-confirm | 0.004899 | 0.013933 | 0.013933 |
| 64 | 0.1 | observed-confirm | 0.004899 | 0.014364 | 0.014364 |
| 64 | 0.1 | race | 0.0062181 | 0.0091975 | 0.0091975 |

For racing policies the last two columns both report the empirical incumbent; there is no larger posterior-mean search in those policies.

## Model and recommendation-search diagnostics

| History | Noise SD | Model | RMSE | Top RMSE | Latent coverage | Top latent coverage | Observation coverage | Search gap | Larger-search gap |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.03 | forest | 0.16106 | 0.03263 | 0.93506 | 0.97927 | 0.94702 | 0 | 0 |
| 32 | 0.03 | gp | 0.079336 | 0.016945 | 0.96497 | 1 | 0.95276 | 2.147e-06 | 0 |
| 32 | 0.1 | forest | 0.17983 | 0.073892 | 0.95044 | 0.95854 | 0.96179 | 0 | 0 |
| 32 | 0.1 | gp | 0.11926 | 0.037937 | 0.95264 | 1 | 0.94299 | 1.7234e-05 | 4.6352e-06 |
| 64 | 0.03 | forest | 0.10847 | 0.025377 | 0.94202 | 0.97317 | 0.95117 | 0 | 0 |
| 64 | 0.03 | gp | 0.035859 | 0.011874 | 0.9696 | 1 | 0.96606 | 0 | 1.138e-05 |
| 64 | 0.1 | forest | 0.11828 | 0.071311 | 0.94727 | 0.93537 | 0.9707 | 0 | 0 |
| 64 | 0.1 | gp | 0.079617 | 0.022574 | 0.95251 | 1 | 0.95105 | 0 | 2.0296e-05 |

## Paired recommendation comparisons

Counts are wins/ties/losses with a 1e-12 tie tolerance.

| Calls | Noise SD | Comparison | Wins/ties/losses |
| ---: | ---: | --- | --- |
| 32 | 0.03 | gp vs forest | 11/0/1 |
| 32 | 0.03 | model-confirm vs observed-confirm | 0/12/0 |
| 32 | 0.03 | model-confirm vs gp | 2/0/10 |
| 32 | 0.1 | gp vs forest | 12/0/0 |
| 32 | 0.1 | model-confirm vs observed-confirm | 0/12/0 |
| 32 | 0.1 | model-confirm vs gp | 1/0/11 |
| 64 | 0.03 | gp vs forest | 10/0/2 |
| 64 | 0.03 | model-confirm vs observed-confirm | 3/6/3 |
| 64 | 0.03 | model-confirm vs gp | 6/0/6 |
| 64 | 0.1 | gp vs forest | 11/0/1 |
| 64 | 0.1 | model-confirm vs observed-confirm | 5/5/2 |
| 64 | 0.1 | model-confirm vs gp | 2/0/10 |

## Archived-history audit

Refit the archived 64-observation histories for seeds 301–308, with their original uniform-noise variance. These are discovery data, excluded from fresh validation. The audit uses the current Cholesky and independent recommendation searches; it does not recreate the earlier solver population or claim a bitwise replay of its recommendations. The CSV observation-coverage field uses a Gaussian probe with the same variance and is not a coverage claim for the original uniform noise; only latent diagnostics are shown here.

| Model | RMSE | Top RMSE | Latent coverage | Small-search regret | Larger-search regret | Reference-pool regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gp | 0.030422 | 0.014766 | 0.98657 | 0.0045391 | 0.004078 | 0.0045391 |
| forest | 0.10978 | 0.026596 | 0.95789 | 0.0039851 | 0.0038699 | 0.0033663 |

The archived GP has lower prediction error but its posterior-minimum recommendations still have higher true regret than the forest on these histories. More thorough mean search does not reverse that result. Latent coverage is high, so these cases do not establish a general overconfidence correction. Prediction accuracy, calibration, and decision regret measure different properties.

The initial single-evaluation-design runs (seeds 1001–1012) are retained under discovery/. They identified over-allocation of initial configurations when racing repeats measurements. The final protocol reuses LatinHypercubeInitialDesign.measurementsPerConfiguration=3 and reserves confirmation cost before sizing the design; no model hyperparameters were selected from the final seeds.

## Decisions and limits

Keep model-based confirmation and independent recommendation search opt-in. A smaller acquisition-search gap does not establish a better true recommendation. Calibration conditions on fitted hyperparameters and model assumptions; this study does not supply a universal variance correction or integrate hyperparameter uncertainty. The two noise levels share one smooth two-dimensional objective, so validation is narrow.

## Reproduce

```sh
./gradlew refineNoisyBo
./gradlew auditLegacyNoisyBo
python3 scripts/summarize_noisy_bo.py
```

Use `-Pnoisy.seeds=911` for development smoke runs, which cannot generate the validation report.
