#!/usr/bin/env python3
"""Archive fresh noisy BO validation without tuning defaults on the validation seeds."""
import csv
import gzip
import hashlib
import io
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path

root = Path('build/reports/noisy-bo')
out = Path('benchmarks/noisy-bo')
out.mkdir(parents=True, exist_ok=True)


def read(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def archive(name, rows):
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    if name.endswith('.gz'):
        (out / name).write_bytes(gzip.compress(text.getvalue().encode(), mtime=0))
    else:
        (out / name).write_text(text.getvalue())


outcomes = read(root / 'outcomes.csv')
diagnostics = read(root / 'diagnostics.csv')
assert len(outcomes) == 240 and len(diagnostics) == 96
assert {int(r['seed']) for r in outcomes} == set(range(1301, 1313))
assert all(int(r['calls']) == int(r['budget']) for r in outcomes)
traces = []
for row in outcomes:
    rows = read(root / f"trace-{row['seed']}-{row['budget']}-{row['sd']}-{row['algorithm']}.csv")
    assert len(rows) == int(row['calls'])
    traces.extend(rows)
archive('outcomes.csv', outcomes)
archive('diagnostics.csv', diagnostics)
archive('traces.csv.gz', traces)


def median(rows, field):
    return stats.median(float(r[field]) for r in rows)


def fmt(value):
    return f'{value:.5g}'


lines = ['# Noisy BO refinement', '',
         'Validation uses fresh seeds 1301–1312; development seeds 911–914 are excluded. '
         'No GP/forest/search defaults were selected from these validation results. '
         'The lifecycle fix makes model inspection independent of future suggestion randomness.', '',
         '## Protocol', '',
         '- Objective: `1 + (x - .31)^2 + 3(y - .72)^2`, with independent Gaussian measurement noise '
         'of standard deviation .03 or .1. Every observation supplies its known noise variance. '
         'Noise comes from Kumulant’s normal sampler with a reproducible configuration/measurement-key seed.',
         '- Budgets are 32/64 external measurements including all racing and confirmation calls. '
         'All 240 runs exhaust their budget; the archive contains 11,520 raw evaluations and coordinates.',
         '- GP uses the unchanged default Matérn model and 24 likelihood candidates. Forest uses ten '
         'bootstrapped trees, random thresholds, both features, and ensemble disagreement. '
         'Responses are standardized; the objective is minimized. All proposals obey Klause feasibility.',
         '- Ordinary GP/forest runs use the same budget-aware Latin-hypercube design and .2 independent '
         'exploration. Acquisition search has 64 initial candidates, four starts, four steps, eight neighbors. '
         'Recommendation compares that search with 1,024 initial candidates on the same cached model.',
         '- Racing initial designs use the existing three-measurements-per-configuration allowance, capped against the search budget after reserving confirmation calls. Racing uses four search keys, a one-measurement initial incumbent/challenger, and a two-standard-error '
         'rejection allowance. Confirmation reserves nine calls: three finalists on three fresh keys. '
         'Observed/model-confirm share identical search histories; only the frozen shortlist ranking changes. '
         'The final winner uses only fresh responses. Plain race spends all calls on search/racing.',
         '- Diagnostics use 4,096 fresh query points plus history and returned recommendations for a finite '
         'posterior-mean reference. The reference includes the returned candidates and is not a global optimum. '
         'Truth is used only for evaluation/top-decile diagnostics, never fitting or finalist ranking. '
         'Observation coverage includes known measurement variance; latent coverage excludes it. '
         'Forest coverage uses a Gaussian approximation to ensemble dispersion.',
         '- Prediction losses use Kumulant MseLossStat; means/coverage use MeanStat. '
         'Recommendation search consumes additional computation, with no external observations. '
         'No wall-clock or universal performance claims are made.', '',
         '## Recommendation and evaluated-point regret', '',
         'Values are medians over twelve paired seeds; lower is better.', '',
         '| Calls | Noise SD | Policy | Best evaluated regret | Recommendation regret | Larger-search regret |',
         '| ---: | ---: | --- | ---: | ---: | ---: |']
groups = defaultdict(list)
for row in outcomes:
    groups[int(row['budget']), float(row['sd']), row['algorithm']].append(row)
for (budget, sd, method), rows in sorted(groups.items()):
    assert len(rows) == 12
    lines.append(f'| {budget} | {sd} | {method} | {fmt(median(rows,"best_regret"))} | '
                 f'{fmt(median(rows,"recommendation_regret"))} | {fmt(median(rows,"large_regret"))} |')
lines += ['', 'For racing policies the last two columns both report the empirical incumbent; '
          'there is no larger posterior-mean search in those policies.', '',
          '## Model and recommendation-search diagnostics', '',
          '| History | Noise SD | Model | RMSE | Top RMSE | Latent coverage | Top latent coverage | Observation coverage | Search gap | Larger-search gap |',
          '| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
groups = defaultdict(list)
for row in diagnostics:
    groups[int(row['history']), float(row['sd']), row['algorithm']].append(row)
for (history, sd, method), rows in sorted(groups.items()):
    fields = ['rmse', 'top_rmse', 'latent_coverage', 'top_latent_coverage', 'observation_coverage', 'small_gap', 'large_gap']
    lines.append(f'| {history} | {sd} | {method} | ' + ' | '.join(fmt(median(rows,f)) for f in fields) + ' |')
lines += ['', '## Paired recommendation comparisons', '',
          'Counts are wins/ties/losses with a 1e-12 tie tolerance.', '',
          '| Calls | Noise SD | Comparison | Wins/ties/losses |', '| ---: | ---: | --- | --- |']
indexed = {(r['seed'], r['budget'], r['sd'], r['algorithm']): r for r in outcomes}
for budget in (32, 64):
    for sd in (0.03, 0.1):
        for left, right, field in [('gp', 'forest', 'recommendation_regret'),
                                   ('model-confirm', 'observed-confirm', 'recommendation_regret'),
                                   ('model-confirm', 'gp', 'recommendation_regret')]:
            ds = [float(indexed[str(s), str(budget), str(sd), right][field]) -
                  float(indexed[str(s), str(budget), str(sd), left][field]) for s in range(1301, 1313)]
            counts = (sum(d > 1e-12 for d in ds), sum(abs(d) <= 1e-12 for d in ds), sum(d < -1e-12 for d in ds))
            lines.append(f'| {budget} | {sd} | {left} vs {right} | {counts[0]}/{counts[1]}/{counts[2]} |')
legacy = read(root / 'legacy/diagnostics.csv')
assert len(legacy) == 16
archive('legacy_diagnostics.csv', legacy)
lines += ['', '## Archived-history audit', '',
          'Refit the archived 64-observation histories for seeds 301–308, with their original '
          'uniform-noise variance. These are discovery data, excluded from fresh validation. '
          'The audit uses the current Cholesky and independent recommendation searches; it does not '
          'recreate the earlier solver population or claim a bitwise replay of its recommendations. '
          'The CSV observation-coverage field uses a Gaussian probe with the same variance and is '
          'not a coverage claim for the original uniform noise; only latent diagnostics are shown here.', '',
          '| Model | RMSE | Top RMSE | Latent coverage | Small-search regret | Larger-search regret | Reference-pool regret |',
          '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for model in ('gp', 'forest'):
    rows = [r for r in legacy if r['algorithm'] == model]
    fields = ['rmse', 'top_rmse', 'latent_coverage', 'small_regret', 'large_regret', 'pool_regret']
    lines.append('| ' + model + ' | ' + ' | '.join(fmt(median(rows, field)) for field in fields) + ' |')
lines += ['', 'The archived GP has lower prediction error but its posterior-minimum recommendations '
          'still have higher true regret than the forest on these histories. More thorough mean search '
          'does not reverse that result. Latent coverage is high, so these cases do not establish a '
          'general overconfidence correction. Prediction accuracy, calibration, and decision regret '
          'measure different properties.', '',
          'The initial single-evaluation-design runs (seeds 1001–1012) are retained under discovery/. '
          'They identified over-allocation of initial configurations when racing repeats measurements. '
          'The final protocol reuses LatinHypercubeInitialDesign.measurementsPerConfiguration=3 '
          'and reserves confirmation cost before sizing the design; no model hyperparameters were '
          'selected from the final seeds.']
lines += ['', '## Decisions and limits', '',
          'Keep model-based confirmation and independent recommendation search opt-in. '
          'A smaller acquisition-search gap does not establish a better true recommendation. '
          'Calibration conditions on fitted hyperparameters and model assumptions; this study does not '
          'supply a universal variance correction or integrate hyperparameter uncertainty. '
          'The two noise levels share one smooth two-dimensional objective, so validation is narrow.', '',
          '## Reproduce', '', '```sh', './gradlew refineNoisyBo', './gradlew auditLegacyNoisyBo', 'python3 scripts/summarize_noisy_bo.py', '```', '',
          'Use `-Pnoisy.seeds=911` for development smoke runs, which cannot generate the validation report.']
(out / 'report.md').write_text('\n'.join(lines) + '\n')
(out / 'protocol.json').write_text(json.dumps({
    'validation_seeds': list(range(1301, 1313)), 'development_seeds': list(range(911, 915)), 'protocol_discovery_seeds': list(range(1001, 1013)),
    'benchmark_sha256': hashlib.sha256(Path('src/jvmTest/kotlin/com/eignex/combo/bo/NoisyBoBenchmark.kt').read_bytes()).hexdigest(),
    'build_sha256': hashlib.sha256(Path('build.gradle.kts').read_bytes()).hexdigest(),
    'runs': len(outcomes), 'evaluations': len(traces),
}, indent=2) + '\n')
print(out / 'report.md')
