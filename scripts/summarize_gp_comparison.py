#!/usr/bin/env python3
"""Archive fresh GP validation seeds and compare model error, search quality, and regret."""
import csv
import gzip
import io
import statistics as stats
from collections import defaultdict
from pathlib import Path

source = Path('build/reports/gp-comparison')
output = Path('benchmarks/gp-comparison')
output.mkdir(exist_ok=True)


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def archive(name, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    data = stream.getvalue().encode()
    if name.endswith('.gz'):
        (output / name).write_bytes(gzip.compress(data, mtime=0))
    else:
        (output / name).write_bytes(data)


def median(rows, field):
    return stats.median(float(row[field]) for row in rows)


def summary(rows, field):
    values = sorted(float(row[field]) for row in rows)
    quartiles = stats.quantiles(values, n=4, method='inclusive')
    return f'{stats.median(values):.3g} [{quartiles[0]:.3g}, {quartiles[2]:.3g}]'


models = [r for r in read(source / 'models.csv') if 401 <= int(r['seed']) <= 408]
search = [r for r in read(source / 'search.csv') if 401 <= int(r['seed']) <= 408]
outcomes = [r for r in read(source / 'outcomes.csv') if 301 <= int(r['seed']) <= 308]
assert len(models) == 128 and len(search) == 256 and len(outcomes) == 192
traces = []
for path in sorted(source.glob('trace-*.csv')):
    rows = read(path)
    if rows and 301 <= int(rows[0]['seed']) <= 308:
        assert len(rows) == int(rows[0]['budget'])
        traces.extend(rows)
assert len(traces) == 9216
for name, rows in [('models.csv', models), ('search.csv', search), ('outcomes.csv', outcomes), ('traces.csv.gz', traces)]:
    archive(name, rows)

by_outcome = {(r['problem'], int(r['budget']), r['algorithm'], r['seed']): r for r in outcomes}
deterministic = [r for r in outcomes if r['algorithm'] == 'gp' and r['problem'] != 'noisy']
def deterministic_wins(other, field):
    return sum(float(r[field]) < float(by_outcome[r['problem'], int(r['budget']), other, r['seed']][field]) - 1e-12
               for r in deterministic)

noisy_gp = [r for r in outcomes if r['problem'] == 'noisy' and r['budget'] == '64' and r['algorithm'] == 'gp']
noisy_forest = [r for r in outcomes if r['problem'] == 'noisy' and r['budget'] == '64' and r['algorithm'] == 'forest']

lines = ['# Continuous GP comparison', '',
         'The batch forest extraction preserves the existing 3,840-row quadratic diagnostic trace '
         'byte-for-byte. These additional experiments compare GP/EI, forest/EI, and random search '
         'without changing the forest defaults. Frozen-history seeds are 401–408; whole-loop seeds '
         'are 301–308. Development seed 91 and preliminary seeds 101–108/201–208 are excluded. '
         'No model settings were selected from these final validation results.', '',
         '## Findings', '',
         f'Across the {len(deterministic)} paired deterministic cases, GP improves best evaluated regret '
         f'in {deterministic_wins("forest", "best_regret")} cases against forest and '
         f'{deterministic_wins("random", "best_regret")} against random search. Recommendation regret '
         f'improves in {deterministic_wins("forest", "recommendation_regret")} and '
         f'{deterministic_wins("random", "recommendation_regret")} cases respectively. The smooth '
         'quadratic and log-coordinate objectives show especially large model-error reductions. '
         'This supports a GP option for these small continuous problems, without changing the forest default.', '',
         'Noise remains a limitation: at 64 calls, median recommendation regret is '
         f'{median(noisy_gp, "recommendation_regret"):.3g} for GP and '
         f'{median(noisy_forest, "recommendation_regret"):.3g} for forest. Paired recommendations '
         'remain mixed, as shown below. Better discovery does not guarantee a better noisy '
         'recommendation. Noise and recommendation calibration remain a refinement target.', '',
         'The fixed-history search tables also show nonzero GP acquisition gaps even when '
         'prediction error is small. Larger candidate pools sometimes help, but these results '
         'do not select a new default search budget.', '',
         '## Protocol', '',
         '- Four continuous objectives have true minimum 1: a shifted quadratic, a multimodal '
         'squared-sine objective, a noisy anisotropic two-dimensional quadratic, and a quadratic '
         'in normalized log coordinates over a variable bounded by 1e-6 and 1.',
         '- Both models see the same frozen histories (16 or 64 points) and 4,096 held-out points. '
         'Responses are standardized from training data only. Known noise variance is 0 for '
         'deterministic objectives and 0.05²/3 for the noisy objective’s uniform synthetic noise. '
         'Truth is used for diagnostics only.',
         '- GP uses the default Matérn-five-halves model, learned per-coordinate length scales '
         'and signal variance, and 24 bounded marginal-likelihood search candidates. Row noise '
         'is supplied in these benchmarks; inference of unknown noise is covered by unit tests. '
         'Its uncertainty conditions on fitted parameters. Forest uses ten bootstrapped randomized '
         'all-feature trees and across-tree disagreement. These are model assumptions, not coverage guarantees.',
         '- Whole-loop budgets are 32 or 64 external evaluations. All three algorithms use the same '
         'seeded Latin-hypercube start (ten points per dimension capped at 25% of budget). '
         'Afterward, random search uses independent uniform draws in encoded geometry; model-based '
         'methods interleave the same independent exploration with probability 0.2. Each generated '
         'point is checked by Klause. EI search uses 64 initial candidates, four starts, four local '
         'steps, and eight neighbors for both models.',
         '- Both models use Gaussian EI on standardized raw responses through the shared acquisition '
         'loop. This compares forest/EI with GP/EI; it is not an external SMAC facade comparison. '
         'Best regret uses the noiseless reference of evaluated points. Recommendation regret uses '
         'model-mean search for GP/forest and best observed response for random search; recommendation '
         'truth is diagnostic and is not fed back or charged as an additional evaluation.',
         '- Values are median [25th percentile, 75th percentile] over eight seeds. These small '
         'synthetic experiments do not establish a universal ranking or wall-clock performance.', '',
         '## Frozen model quality', '',
         'Top-region RMSE uses the best 10% of held-out true objective values for reporting only. '
         'Coverage is the fraction of latent truth within two predicted standard deviations.', '',
         '| Problem | History | Model | RMSE | Top-region RMSE | Two-SD coverage | Mean SD |',
         '| --- | ---: | --- | --- | --- | --- | --- |']
groups = defaultdict(list)
for row in models:
    groups[row['problem'], int(row['history']), row['algorithm']].append(row)
for (problem, count, model), rows in sorted(groups.items()):
    assert len(rows) == 8
    lines.append(f'| {problem} | {count} | {model} | {summary(rows, "rmse")} | '
                 f'{summary(rows, "top_rmse")} | {summary(rows, "coverage_two_sd")} | {summary(rows, "mean_sd")} |')
lines += ['', '## Acquisition search on fixed histories', '',
          'The reference is the largest acquisition value over the fixed 4,096-point pool, not a '
          'global maximum. The initial pools of 64 and 256 are nested prefixes. Local search may '
          'exceed the finite reference; gaps below zero are clipped for this diagnostic. A larger '
          'acquisition value is not necessarily better true objective value.', '',
          '| Problem | History | Model | Initial pool | Median acquisition gap | Median selected true regret |',
          '| --- | ---: | --- | ---: | ---: | ---: |']
groups = defaultdict(list)
for row in search:
    row['gap'] = max(0.0, float(row['reference_acquisition']) - float(row['selected_acquisition']))
    row['regret'] = float(row['selected_truth']) - 1.0
    groups[row['problem'], int(row['history']), row['algorithm'], int(row['pool'])].append(row)
for (problem, count, model, pool), rows in sorted(groups.items()):
    lines.append(f'| {problem} | {count} | {model} | {pool} | {median(rows, "gap"):.3g} | {median(rows, "regret"):.3g} |')
lines += ['', '## Equal external budgets', '',
          '| Problem | Budget | Algorithm | Best evaluated regret | Recommendation regret |',
          '| --- | ---: | --- | --- | --- |']
groups = defaultdict(list)
for row in outcomes:
    groups[row['problem'], int(row['budget']), row['algorithm']].append(row)
for (problem, budget, algorithm), rows in sorted(groups.items()):
    assert len(rows) == 8
    lines.append(f'| {problem} | {budget} | {algorithm} | {summary(rows, "best_regret")} | '
                 f'{summary(rows, "recommendation_regret")} |')
lines += ['', '## Paired comparisons', '',
          'GP wins/ties/losses count lower regret on the same seed. Differences within 1e-12 are ties.', '',
          '| Problem | Budget | Comparator | Best evaluated GP W/T/L | Recommendation GP W/T/L |',
          '| --- | ---: | --- | --- | --- |']
for problem, budget in sorted({(r['problem'], int(r['budget'])) for r in outcomes}):
    gp = {r['seed']: r for r in groups[problem, budget, 'gp']}
    for other in ('forest', 'random'):
        counts = []
        for field in ('best_regret', 'recommendation_regret'):
            difference = [float(gp[r['seed']][field]) - float(r[field]) for r in groups[problem, budget, other]]
            counts.append(f'{sum(d < -1e-12 for d in difference)}/{sum(abs(d) <= 1e-12 for d in difference)}/{sum(d > 1e-12 for d in difference)}')
        lines.append(f'| {problem} | {budget} | {other} | {counts[0]} | {counts[1]} |')
lines += ['', '## Scope and next refinements', '',
          'Keep GP/EI as an explicit alternative for small continuous histories; retain the forest '
          'defaults for existing users. Read the frozen model and fixed-history acquisition tables '
          'together before attributing an optimization difference to search or uncertainty. '
          'Mixed-type/conditional kernels, censored outcomes, multi-fidelity scheduling, and joint '
          'batch EI are unvalidated or unimplemented here. Exact GP fits retain history and scale '
          'cubically; coherent lazy function draws also retain their distinct queried points.', '',
          'The GP follows [GPML chapter 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf). '
          'Koblas supplies numerical factorization and solves; all GP model and optimization policy '
          'lives in Combo. Kumulant supplies generic summaries and shared tree snapshots.', '',
          '## Reproduce', '', '```sh', './gradlew compareGp',
          'python3 scripts/summarize_gp_comparison.py', '```', '',
          'Use `-Pgp.mode=frozen` or `-Pgp.mode=optimize` to run one stage. '
          '`-Pgp.seeds=91` is a development smoke run and is excluded from this report. '
          'Pinned build dependencies define the numerical runtime. Raw diagnostic rows remain '
          'under build/reports/gp-comparison; the archived CSVs and compressed traces retain all '
          'final validation metrics and external evaluations.', '']
(output / 'report.md').write_text('\n'.join(lines))
print(f'Archived {len(models)} model rows, {len(search)} search rows, {len(outcomes)} outcomes, and {len(traces)} evaluations.')
