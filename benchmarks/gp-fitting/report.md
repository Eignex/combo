# GP fitting and calibration refinement

The combined covariance-weighted mean, log-parameter regularization, and multistart policy
improves recommendation regret in 61/96 fresh pairs against the existing 24-trial fit,
with 34 losses and one tie. Against the equal-budget 96-trial coordinate fit it wins
55/96, loses 40, and ties once. Results vary with the objective and noise condition,
and restart sensitivity remains material. Existing fitting defaults remain unchanged;
the policies are independently configurable and opt-in.

## Ownership and statistical semantics

Kumulant owns `estimateCorrelatedMean(values, covarianceFactor)`: generalized least squares
for a common location with a supplied positive-definite observation covariance. It returns
the estimated location and its variance conditional on that covariance, uses the existing
Kumulant Cholesky whitening solves and Koblas dot products, and never forms a matrix inverse.
The estimator also serves correlated measurements independently of GP or BO use.

Combo profiles the GP's constant mean at each covariance candidate when requested. An
explicit configured mean takes precedence. Conditioning freezes the estimated mean and
covariance; mean-estimation and covariance-estimation uncertainty are not integrated into
the latent posterior. The original empirical-mean policy remains available and default.

Combo also owns GP-specific hyperparameter policies. Regularization adds independent Gaussian
penalties on log covariance parameters, with constants omitted. These are densities in log
coordinates, not log-normal densities in original coordinates: no Jacobian term is added.
Length-scale centers are `0.5 * sqrt(d)` with log deviation 1; the signal-variance center is
1 with log deviation 2. Unknown-noise variance is centered at its initial value with log
deviation 2. Known row variances are fixed and never penalized or optimized. Centers and
deviations are configurable. Default centers assume normalized inputs and standardized responses.
This is a deliberately modest regularizer, not a reproduction of BoTorch's priors.
[BoTorch's dimension-scaled priors](https://botorch.org/docs/models) and
[GPML's mean and hyperprior fitting](https://gaussianprocess.org/gpml/code/matlab/doc/)
provide reference context.

`logMarginalLikelihood` retains its likelihood-only meaning; `logFitScore` adds the penalties.
The posterior remains conditional on a single selected parameter vector. Multistart fitting
splits the total covariance-candidate budget across four starts, including restart points.
The first start uses configured parameters; others independently perturb each log parameter
uniformly within plus/minus log(4) of its initial value. Each start uses the existing bounded
coordinate refinement. The best scoring candidate across all starts is retained, including
the initial fit. A fixed training seed is reproducible; different seeds can select different fits.

Koblas supplies the existing BLAS operations. No Koblas implementation changes are needed.
Combo's dependency pins follow the platform versions used by the updated Kumulant artifact.

## Frozen-history ablations

Use 96 final 32-evaluation histories from the retained GP refinement runs: four objectives,
seeds 1701–1706, known measurement noise SD 0/.03, q=1/4, and lognei-local. Restore reverse
completion order within each batch and retain the original response standardization. Compare:

- baseline24: original empirical mean, no regularizer, 24 coordinate candidates.
- baseline96: the same fit with 96 candidates.
- mean96: covariance-weighted mean with 96 coordinate candidates.
- prior96: empirical mean and log-parameter regularization with 96 coordinate candidates.
- multistart96: empirical mean and four starts, no regularizer, 96 total candidates.
- combined96: covariance-weighted mean, regularization, four starts, 96 total candidates.

Every fit evaluates one initial covariance in addition to its candidate budget. All recommendations
use the same independent search budget: 128 random candidates, four starts, four local steps,
eight neighbors, and observed configurations as anchors. Hold a shared independent 4,096-point
uniform prediction set fixed per history. Kumulant MseLossStat scores prediction errors;
MeanStat scores two-standard-deviation latent coverage. Top-region diagnostics use the best
true decile of the prediction set. Truth is used only for evaluation.

The existing baseline24 recommendations match all 48 overlapping quadratic/ellipsoid results
from the earlier follow-up exactly. This checks the behavior-preserving default and dependency update.

The full [frozen tables](tables.md) show that mean estimation often reduces top-region error,
while regularization and mean estimation change which recommendation is selected. Dividing the
budget across starts weakens some noiseless fits. Larger likelihood budgets alone still worsen
noisy six-dimensional recommendation regret. These are discovery histories, not independent validation.
The pool gap only compares predicted means over a finite shared set and is not a global optimum
certificate. High aggregate coverage, often close to one, is not evidence of precise calibration.

## Restart sensitivity

For the 24 noisy q=4 discovery histories, evaluate multistart96 and combined96 with eight seeded
restart streams per history, giving 384 fits. No configuration is selected using the best of
these streams. Median within-history ranges of true recommendation regret are:

| Problem | Multistart96 | Combined96 |
| --- | --- | --- |
| quadratic2 | 0.000277685 | 0.00165678 |
| branin2 | 0.00317134 | 0.00386267 |
| multimodal3 | 0.011447 | 0.00752185 |
| ellipsoid6 | 0.00938059 | 0.00271819 |

Seed reproducibility does not remove initialization sensitivity. Four shorter refinements can
miss an optimum reached by one longer refinement, even when all candidates share the same bounds.

## Fresh whole-loop validation

Select mean96 and combined96 from frozen diagnostics before running unused seeds 1901–1906.
Compare them with baseline24 and the equal-candidate-budget baseline96 across all four objectives,
SD 0/.03 and q=1/4: 384 runs and 12,288 external evaluations. Each run has 32 evaluations,
eight initial Latin-hypercube observations, the existing .2 exploration setting, qLogNEI with
256 antithetic draws, and the same acquisition and recommendation search budgets. Completion
order is reversed within each batch. Only the fitting policy and its declared budget differ.
Known measurement variances are supplied; these runs do not validate unknown-noise estimation.
The multimodal objective challenges this kernel and data budget; these experiments do not isolate
model misspecification from fitting error. Best-evaluated regret uses latent truth for evaluation,
not a recommendation available to the optimizer.

Median true recommendation regret, aggregating six seeds and both batch sizes:

| Problem | SD | Baseline24 | Baseline96 | Mean96 | Combined96 |
| --- | --- | --- | --- | --- | --- |
| quadratic2 | 0.0 | 2.21087e-06 | 1.79717e-06 | 2.41578e-06 | 1.85048e-06 |
| quadratic2 | 0.03 | 0.00260375 | 0.00159792 | 0.00237646 | 0.00258772 |
| branin2 | 0.0 | 0.000874876 | 0.000134113 | 0.000109336 | 0.000219175 |
| branin2 | 0.03 | 0.00400702 | 0.00590138 | 0.00309458 | 0.00525152 |
| multimodal3 | 0.0 | 0.0237287 | 0.0241123 | 0.0231972 | 0.0239098 |
| multimodal3 | 0.03 | 0.0320834 | 0.0292369 | 0.0311348 | 0.0229367 |
| ellipsoid6 | 0.0 | 0.00106443 | 0.000749484 | 0.000391024 | 0.00096546 |
| ellipsoid6 | 0.03 | 0.0146236 | 0.0238757 | 0.0266415 | 0.0097123 |

Compared with baseline24, mean96 has 50/45/1 recommendation wins/losses/ties and combined96
has 61/34/1. Best-evaluated regret comparisons are 53/35/8 and 53/38/5, respectively.
Against baseline96, combined96 has 55/40/1 recommendation wins/losses/ties and 46/43/7 for
best-evaluated regret. Ties use absolute tolerance 1e-10. Individual objectives have only
six independent seeds; counts and medians are descriptive, not a statistical superiority claim.
The frozen combined policy's promise survives some fresh tests but not all noise conditions.

The report retains per-fit nanoseconds and whole-loop suggestion/recommendation nanoseconds.
They come from a single JVM with shared-host/JIT/order effects and include different amounts of
search work; they are not throughput evidence. Candidate counts, rather than wall-clock equality,
provide the controlled fitting-work budget. No performance claim or timing-based default change is made.

## Reproduction and retained data

From the pinned standalone Combo build:

```sh
./gradlew refineGpFitting -Pfitting.mode=frozen
./gradlew refineGpFitting -Pfitting.mode=sensitivity
./gradlew refineGpFitting -Pfitting.mode=loop -Pfitting.variants=baseline24,baseline96,mean96,combined96
python3 scripts/summarize_gp_fitting.py
```

The Gradle commands write `build/reports/gp-fitting/<mode>/`. The summary script reads retained
archives under `benchmarks/gp-fitting/`; copy or deterministically gzip refreshed outputs there
before regenerating `tables.md`. The archives contain all 576 frozen fits, 384 sensitivity fits,
384 fresh outcomes, and all 12,288 fresh trace rows. `metadata.json` records budgets and provenance.
Analytic tests cover generalized least squares, profiled-mean likelihood, explicit means,
fixed conditioning parameters, known versus unknown noise, regularized scores, seeded fitting,
and zero candidate budgets. Full local gates and PR CI validate the final published dependencies.
