#!/usr/bin/env python3
"""Validate, archive, and summarize the pinned SMAC / Combo comparison (stdlib only)."""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def fmt(value):
    return f'{value:.3g}'


def summarize(directory, archive):
    runs = defaultdict(dict)
    rows = []
    for path in sorted(directory.glob('*.csv')):
        with path.open() as source:
            trace = list(csv.DictReader(source))
        if len(trace) != 32 or [int(r['evaluations']) for r in trace] != list(range(1, 33)):
            raise ValueError(f'{path}: incomplete external budget')
        regrets = [float(r['simple_regret']) for r in trace]
        if any(x < 0 for x in regrets) or any(b > a for a, b in zip(regrets, regrets[1:])):
            raise ValueError(f'{path}: invalid regret trace')
        first = trace[0]
        key = first['problem'], first['algorithm']
        seed = int(first['seed'])
        if seed in runs[key]:
            raise ValueError(f'duplicate run {key} {seed}')
        runs[key][seed] = trace
        rows.extend(trace)
    if len(runs) != 13 or any(set(r) != {3, 11, 19, 29, 43, 61} for r in runs.values()):
        raise ValueError('Expected four problems, three algorithms each, plus noisy Combo racing, with six seeds each')
    with archive.open('w') as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    out = ['# Pinned SMAC comparison', '',
           'SMAC 2.4.1 HPO facade versus the new Combo profile and feasible random search. '
           'Six paired seeds; 32 external objective calls per run, including repetitions. '
           'All costs are minimized and have noiseless optimum one. Lower regret is better.', '',
           '## Discovery regret', '',
           '| Problem | Algorithm | Median @8 | Median @16 | Median @32 | Final IQR |',
           '| --- | --- | ---: | ---: | ---: | --- |']
    for (problem, algorithm), seeds in sorted(runs.items()):
        medians = [statistics.median(float(t[n - 1]['simple_regret']) for t in seeds.values()) for n in (8, 16, 32)]
        final = [float(t[-1]['simple_regret']) for t in seeds.values()]
        q1, _, q3 = statistics.quantiles(final, n=4, method='inclusive')
        out.append(f'| {problem} | {algorithm} | ' + ' | '.join(map(fmt, medians)) + f' | {fmt(q1)}–{fmt(q3)} |')
    out += ['', '## Paired final discovery comparisons', '',
            '| Problem | Combo vs random wins/ties/losses | Combo vs SMAC wins/ties/losses |',
            '| --- | --- | --- |']
    for problem in ('quadratic', 'logarithmic', 'mixed', 'noisy'):
        counts = []
        combo = runs[problem, 'combo-profile']
        for algorithm in ('random', 'smac-2.4.1'):
            differences = [float(combo[s][-1]['simple_regret']) - float(t[-1]['simple_regret'])
                           for s, t in runs[problem, algorithm].items()]
            counts.append('/'.join(str(sum(test(d) for d in differences)) for test in
                                   (lambda d: d < 0, lambda d: d == 0, lambda d: d > 0)))
        out.append(f'| {problem} | ' + ' | '.join(counts) + ' |')
    out += ['', '## Noisy recommendations', '',
            'These use each algorithm’s selected **evaluated** configuration, scored without observation noise '
            'for reporting only. Combo without racing and random search select the lowest measured replicate mean; '
            'racing uses its fully confirmed incumbent; SMAC uses its intensifier incumbent. '
            'This differs from a search over the surrogate mean and may be worse than the best point discovered.', '',
            '| Algorithm | Median recommendation regret @32 | Final IQR | Median distinct configurations |',
            '| --- | ---: | --- | ---: |']
    for algorithm in ('combo-profile', 'combo-racing', 'random', 'smac-2.4.1'):
        traces = runs['noisy', algorithm].values()
        values = [float(t[-1]['recommendation_regret']) for t in traces]
        q1, _, q3 = statistics.quantiles(values, n=4, method='inclusive')
        unique = [len({(r['x'], r['y'], r['count'], r['mode']) for r in t}) for t in traces]
        out.append(f'| {algorithm} | {fmt(statistics.median(values))} | {fmt(q1)}–{fmt(q3)} '
                   f'| {statistics.median(unique):g} |')
    out += ['', '## Method and limits', '',
            '- Full per-evaluation coordinates, measured responses, discovery/regret, recommendation regret, '
            'measurement seeds, and costs are archived in [smac-traces.csv](smac-traces.csv). '
            'The first seed of every Combo case is rerun with an exact non-timing trace assertion. '
            'Timing is diagnostic only: runs overlapped builds and do not establish wall-clock rankings.',
            '- See [reproduction instructions](README.md#pinned-smac-comparison) and the locked Python environment. '
            'SMAC uses its stock HPO facade: Sobol design (10 per dimension, capped at 25% of trials), '
            '20% random challengers, 10 randomized all-feature trees, log-cost EI, 10 local starts, '
            '10,000 initial challengers, and an intensifier with at most 3 measurements per configuration.',
            '- Combo uses a continuous Latin hypercube with the same dimension/budget sizing, 20% independent '
            'feasible exploration, 10 randomized all-feature Kumulant trees, lognormal EI, 256 initial candidates, '
            '10 adaptive local walks, 8 steps per walk, and 12 attempted neighbors per step. '
            'Internal computation budgets and forest statistics differ; equal external budgets do not imply '
            'implementation parity. Kumulant variance is across-tree disagreement; SMAC also uses within-leaf statistics.',
            '- Deterministic objectives are quadratic, logarithmic, and mixed (real, integer, category). '
            'They match the legacy benchmark geometry, shifted to strictly positive costs. The noisy objective '
            'is `1 + (x - .31)^2 + 3*(y - .72)^2` plus a fixed configuration/seed pseudo-noise function '
            'with amplitude .05. The three noisy measurement seeds are read from SMAC’s measurement identities; '
            'Combo never reads its outcomes. Plain Combo and random search use the first identity for each novel '
            'point; racing and SMAC repeat points on matched identities. The oracle is used only for reporting.',
            '- Six seeds and small synthetic objectives do not establish statistical significance or general '
            'superiority. Discovery and final recommendation measure different goals. Racing is optional and '
            'can spend a short budget confirming points at the expense of discovery. Its early rejection uses '
            'paired means with a reproducibly shuffled challenger key order, without confidence bounds. Instance-feature marginalization, censored runtimes, '
            'Hyperband/multi-fidelity promotion, and joint batch acquisition remain absent. '
            'Exact affine feasibility is checked separately by the constrained regression benchmarks.', '']
    return '\n'.join(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('build/reports/smac-comparison'))
    parser.add_argument('--output', type=Path, default=Path('benchmarks/smac-comparison.md'))
    parser.add_argument('--archive', type=Path, default=Path('benchmarks/smac-traces.csv'))
    args = parser.parse_args()
    args.output.write_text(summarize(args.directory, args.archive))
