#!/usr/bin/env python3
"""Archive the refinement experiments and report held-out comparisons without ranking assertions."""
import csv
import gzip
import io
import statistics as stats
from collections import defaultdict
from pathlib import Path

root = Path('build/reports/bo-refinement')
out = Path('benchmarks/refinement')
out.mkdir(exist_ok=True)


def read(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def archive(name, rows):
    destination = io.StringIO()
    writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    if name == 'traces.csv':
        (out / 'traces.csv.gz').write_bytes(gzip.compress(destination.getvalue().encode(), mtime=0))
    else:
        (out / name).write_text(destination.getvalue())


def median(rows, field):
    return stats.median(float(r[field]) for r in rows)


def number(value):
    return f'{value:.4g}'


models = read(root / 'models.csv') + read(root / 'smac-models.csv')
models = [r for r in models if 201 <= int(r['seed']) <= 220]
search = [r for r in read(root / 'search.csv') if 201 <= int(r['seed']) <= 220]
traces = []
for path in sorted(root.glob('*.csv')):
    if path.name.startswith(('frozen-', 'models', 'search', 'smac-')):
        continue
    rows = read(path)
    if rows and 101 <= int(rows[0]['seed']) <= 112:
        if len(rows) != int(rows[0]['budget']):
            raise ValueError(f'incomplete budget: {path}')
        traces.extend(rows)
archive('models.csv', models)
archive('search.csv', search)
archive('traces.csv', traces)
lines = ['# BO refinement cycle', '',
         'Frozen-history diagnostics use 20 seeds (201–220). Whole-loop validation uses 12 different '
         'seeds (101–112), with separate runs budgeted at 32 and 128 external calls. Development seed 91 '
         'is excluded. Lower regret and prediction error are better. No algorithm ranking is asserted by tests.', '',
         '## Decisions', '',
         '- Keep disagreement-only uncertainty as the default. Extra leaf-mean variance widens coverage '
         'without changing fitted means and does not consistently improve recommendation regret.',
         '- Keep racing and confirmation opt-in. Accounting for measurements fixes design allocation, '
         'but guarded rejection still spends too much of a 32-call budget on repeats. Fresh-key confirmation '
         'helps some guarded runs and does not consistently beat ordinary search.',
         '- Do not raise the default candidate pool to 10,000 based on these tests. The quadratic’s median '
         'reference-pool acquisition gap is already zero at 64 candidates. Noisy cases have search misses, '
         'but acquisition maximization and actual objective improvement remain different measurements.', '',
         '## Frozen models', '',
         'Every model receives identical independently sampled raw-coordinate histories and predicts the '
         'same 10,000 held-out points. These diagnostics use raw positive responses, not the log-response '
         'whole-loop profile. Coverage is the fraction of noiseless reference targets within two predicted '
         'standard deviations; it is descriptive coverage, not a confidence guarantee. Top-region metrics '
         'use the best 10% of held-out true costs for reporting only. Known noise is supplied '
         'to Combo’s leaf-mean estimator, while SMAC and disagreement mode do not use it.', '',
         '| Problem | History | Model | Median RMSE | Median coverage | Median SD | Top-region RMSE | Top-region coverage |',
         '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
groups = defaultdict(list)
for row in models:
    groups[row['problem'], int(row['history']), row['uncertainty']].append(row)
if len(groups) != 12 or any(len(g) != 20 for g in groups.values()):
    raise ValueError('expected 20 seeds for each of 12 frozen model groups')
for (problem, history, model), rows in sorted(groups.items()):
    lines.append(f'| {problem} | {history} | {model} | {number(median(rows, "rmse"))} | '
                 f'{number(median(rows, "coverage_two_sd"))} | {number(median(rows, "mean_sd"))} | '
                 f'{number(median(rows, "top_rmse"))} | {number(median(rows, "top_coverage_two_sd"))} |')
lines += ['', '## Acquisition search at fixed histories', '',
          'Only the initial candidate budget varies; each run retains four adaptive starts, four local '
          'steps, and eight attempted neighbors. Initial pools are nested prefixes of the same 10,000 '
          'feasible held-out points. The reference maximum is over that finite pool, not a global optimum. '
          'Local search can exceed it. The gap is clipped at zero only for this diagnostic.', '',
          '| Problem | History | Uncertainty | Candidates | Median acquisition gap | Median selected true cost |',
          '| --- | ---: | --- | ---: | ---: | ---: |']
groups = defaultdict(list)
for row in search:
    groups[row['problem'], int(row['history']), row['uncertainty'], int(row['candidates'])].append(row)
for (problem, history, model, count), rows in sorted(groups.items()):
    if len(rows) != 20:
        raise ValueError('incomplete search group')
    gap = stats.median(max(0, float(r['reference_acquisition']) - float(r['selected_acquisition'])) for r in rows)
    lines.append(f'| {problem} | {history} | {model} | {count} | {number(gap)} | '
                 f'{number(median(rows, "selected_truth"))} |')
finals = [r for r in traces if r['step'] == r['budget']]
archive('outcomes.csv', finals)
groups = defaultdict(list)
for row in finals:
    groups[row['problem'], int(row['budget']), row['algorithm']].append(row)
if len(groups) != 28 or any(len(g) != 12 for g in groups.values()):
    raise ValueError('expected 12 validation seeds for each of 28 whole-loop groups')
lines += ['', '## Whole-loop validation', '',
          'All model-driven variants use the same bounded search profile: ten randomized all-feature trees, '
          'log-response EI, 64 initial candidates, four starts, four steps, eight neighbors, and 20% '
          'independent exploration. This is smaller than the original 256-candidate/ten-start SMAC '
          'comparison and does not replace those historical results. The conditional problem combines '
          'a real variable, category, and optional integer subspace. Every proposal is checked in source coordinates.', '',
          '| Problem | Calls | Algorithm | Median discovery regret | Median recommendation regret | Recommendation IQR | Median configurations |',
          '| --- | ---: | --- | ---: | ---: | --- | ---: |']
for (problem, budget, algorithm), rows in sorted(groups.items()):
    values = [float(r['recommendation_regret']) for r in rows]
    q1, _, q3 = stats.quantiles(values, n=4, method='inclusive')
    lines.append(f'| {problem} | {budget} | {algorithm} | {number(median(rows, "best_regret"))} | '
                 f'{number(stats.median(values))} | {number(q1)}–{number(q3)} | '
                 f'{median(rows, "distinct_configurations"):g} |')
lines += ['', '## Paired recommendation comparisons', '',
          '| Problem | Calls | Candidate | Reference | Wins / ties / losses |',
          '| --- | ---: | --- | --- | --- |']
for problem in ('quadratic', 'conditional', 'noisy'):
    for budget in (32, 128):
        comparisons = [('leaf', 'disagreement')]
        if problem == 'noisy':
            comparisons += [('budgeted-race', 'legacy-race'), ('guarded-race', 'budgeted-race'),
                            ('confirmed-race', 'guarded-race'), ('confirmation', 'disagreement')]
        for candidate, reference in comparisons:
            left = {int(r['seed']): r for r in groups[problem, budget, candidate]}
            right = {int(r['seed']): r for r in groups[problem, budget, reference]}
            diffs = [float(left[s]['recommendation_regret']) - float(right[s]['recommendation_regret']) for s in left]
            counts = [sum(d < -1e-12 for d in diffs), sum(abs(d) <= 1e-12 for d in diffs), sum(d > 1e-12 for d in diffs)]
            lines.append(f'| {problem} | {budget} | {candidate} | {reference} | ' + ' / '.join(map(str, counts)) + ' |')
lines += ['', '## Policies and limits', '',
          '- `disagreement` retains the existing forest uncertainty. `leaf` adds approximate leaf-mean '
          'sampling variance using original observation units. Their frozen fitted means are identical. '
          'Leaf-mean uncertainty can overlap tree disagreement and can overcover; it remains opt-in.',
          '- `legacy-race` retains full-plan initial incumbent evaluation and paired-mean rejection. '
          '`budgeted-race` only divides the initial configuration allowance by the three planned measurement '
          'keys. `guarded-race` additionally starts incumbents with two keys and uses a two-standard-error '
          'allowance before early rejection. Repeated error-band comparisons are not sequential significance tests.',
          '- `confirmed-race` adds a frozen top-three shortlist and a fresh-key final comparison. '
          '`confirmation` applies that final comparison to ordinary one-measurement search. '
          'Each finalist receives two fresh measurements at budget 32, or eight at budget 128; '
          'search stops at 26 or 104 calls. The final recommendation uses only fresh-key outcomes. '
          'Every confirmation call counts toward the same external budget and trains once.',
          '- Noisy costs are `1 + (x-.31)^2 + 3*(y-.72)^2` plus a fixed seed/configuration pseudo-noise '
          'function of amplitude .05; supplied variance is .0025/3. Different measurement keys produce '
          'paired reproducible outcomes. This is one synthetic noise process; heavy tails, drifting noise, '
          'and varying instance difficulty need further validation. Oracle costs are used for reporting only.',
          '- The [pinned SMAC 2.4.1 sklearn backend](https://github.com/automl/SMAC3/blob/v2.4.1/smac/model/random_forest/random_forest.py) uses variance across tree predictions. The previous '
          'report’s claim that it adds within-leaf variance was incorrect and has been corrected. '
          'The frozen reference uses SMAC’s HPO tree settings with log_y disabled to match these raw responses; '
          'encoding is identical, so differences in its ConfigSpace categorical handling are absent here.',
          '- The first smoke run with an optional real subspace was interrupted after exact LP certification '
          'dominated runtime. The final conditional case uses an optional integer subspace. This study does '
          'not establish performance on optional-real constrained spaces or compare wall-clock speed.',
          '- [Final outcomes](outcomes.csv), aggregate diagnostics, and [compressed complete traces](traces.csv.gz) are archived beside this report. '
          'Frozen numeric datasets remain under `build/reports/bo-refinement/` and are reproducible from '
          'the runner and seed lists. See [reproduction instructions](../README.md#refinement-cycle).', '']
(out / 'report.md').write_text('\n'.join(lines))
