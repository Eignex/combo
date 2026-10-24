#!/usr/bin/env python3
"""Summarize the quadratic replication/plateau ablation emitted by ForestEiDiagnosticsTest."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(path):
    runs = defaultdict(list)
    with path.open() as source:
        for row in csv.DictReader(source):
            runs[row['group'], row['variant'], row['target'], row['seed']].append(row)
    regrets = {}
    for key, rows in runs.items():
        if len(rows) != 32 or {int(row['step']) for row in rows} != set(range(1, 33)):
            raise ValueError(f'incomplete or duplicate evaluation steps: {key}')
        regrets[key] = min(-float(row['reward']) for row in rows)
    lines = [
        '# Why forest/EI lost to random search on the quadratic', '',
        'The traced implementation repeatedly evaluated completed points. Forest predictions are',
        'piecewise constant, so many candidates shared exactly equal EI. Candidate-order ties favored',
        'history anchors, which appeared before newly generated points. Bootstrap-tree disagreement',
        'can also remain positive at an evaluated point, so Gaussian EI can assign positive scores',
        'to an outcome already known exactly. This uncertainty proxy does not enforce noiseless',
        'posterior conditioning. Repeating an unchanged deterministic point consumed objective',
        'budget without adding spatial information.', '',
        'The fix excludes completed assignments for deterministic outcomes and uses seeded random',
        'tie-breaking with bounded plateau exploration. Excluded anchors remain usable for local',
        'geometry, with restarts when an excluded maximizer yields no eligible incumbent. Each',
        'distinct source assignment is scored once per search. Noisy replication remains configurable.', '',
        'SMAC also handles forest plateaus explicitly and randomizes equal-valued configurations:',
        '[local-search implementation](https://github.com/automl/SMAC3/blob/main/smac/acquisition/maximizer/local_search.py).', '',
        '## Ablation', '',
        'Every run evaluates `-(x - target)^2` on [0, 1] exactly 32 times. Forest configuration is',
        'unchanged: eight trees, five initial observations, 10% random exploration, 32 initial',
        'candidates, six local steps, and twelve neighbors. The trace wrapper only records existing',
        'scores; it does not consume randomness or add acquisition calls.', '',
        '- `baseline`: completed-point repeats allowed, first-candidate ties (previous behavior).',
        '- `novel`: exclude completed points, first-candidate ties.',
        '- `ties`: repeats allowed, seeded random ties.',
        '- `both`: exclude completed points and randomize ties (current deterministic defaults).',
        '- `random`: unchanged feasible random-search baseline.', '',
        'Discovery uses target 0.73 and seeds 3, 11, 19, 29, 43, 61. Validation uses targets',
        '0.17, 0.43, 0.86 and different seeds 2, 5, 7, 13, 23, 37 (18 runs per variant).',
        'Lower regret is better. IQR uses inclusive 25th–75th percentiles.', '',
        '| Cases | Variant | Median final regret | IQR | Repeated evaluations / total | Median ties at scored model maximum |',
        '|---|---|---:|---:|---:|---:|',
    ]
    variants = ('baseline', 'novel', 'ties', 'both', 'random')
    for group, count in (('discovery', 6), ('validation', 18)):
        for variant in variants:
            keys = [key for key in runs if key[:2] == (group, variant)]
            if len(keys) != count:
                raise ValueError(f'expected {count} runs: {(group, variant)}')
            values = [regrets[key] for key in keys]
            quartiles = statistics.quantiles(values, n=4, method='inclusive')
            rows = [row for key in keys for row in runs[key]]
            repeats = sum(row['repeated'] == 'true' for row in rows)
            model_ties = [int(row['tied_maximum']) for row in rows if row['model_driven'] == 'true']
            ties = f'{statistics.median(model_ties):g}' if model_ties else '—'
            lines.append(f'| {group} | {variant} | {statistics.median(values):.6g} | '
                         f'{quartiles[0]:.6g}–{quartiles[2]:.6g} | {repeats} / {len(rows)} | {ties} |')
    lines += ['', 'Model maxima include excluded geometric anchors. Random-design rounds are not labeled EI rounds.', '',
              '## Paired final-regret comparisons', '',
              '| Cases | Both versus baseline (wins / ties / losses) | Both versus random |', '|---|---:|---:|']
    for group in ('discovery', 'validation'):
        cells = []
        keys = [key for key in runs if key[:2] == (group, 'both')]
        for other in ('baseline', 'random'):
            differences = [regrets[key] - regrets[group, other, key[2], key[3]] for key in keys]
            wins = sum(delta < -1e-12 for delta in differences)
            ties = sum(abs(delta) <= 1e-12 for delta in differences)
            losses = sum(delta > 1e-12 for delta in differences)
            cells.append(f'{wins} / {ties} / {losses}')
        lines.append(f'| {group} | {cells[0]} | {cells[1]} |')
    lines += ['', '## What remains', '',
              'These changes remove demonstrated selection waste; they do not turn forest disagreement',
              'into a calibrated posterior or give CART a smooth-function prior. Individual losses remain.',
              'The broader [five-objective results](bo-regret.md) remain part of validation, including the',
              'noisy case. Its paired outcomes still include losses to random search even with a lower median.',
              'No objective-specific optimum, variance floor, or quadratic interpolation was',
              'added to the algorithm. These small synthetic samples do not establish statistical significance.', '',
              'Generated by `python3 scripts/summarize_forest_ei.py`; see [reproduction instructions](README.md).', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('build/reports/bo-diagnostics/quadratic.csv'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = summarize(args.input)
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end='')
