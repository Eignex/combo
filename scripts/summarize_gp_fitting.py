#!/usr/bin/env python3
"""Summarize retained Kumulant-scored fitting diagnostics and paired BO outcomes."""
import csv
import gzip
from pathlib import Path
from statistics import median

ROOT = Path('benchmarks/gp-fitting')
VARIANTS = ('baseline24', 'baseline96', 'mean96', 'prior96', 'multistart96', 'combined96')


def rows(name):
    with gzip.open(ROOT / name, 'rt') as stream:
        return list(csv.DictReader(stream))


def table(headers, records):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, record)) + ' |' for record in records])


def summary():
    frozen = rows('frozen-scores.csv.gz')
    records = []
    for problem in ('quadratic2', 'branin2', 'multimodal3', 'ellipsoid6'):
        for sd in ('0.0', '0.03'):
            for variant in VARIANTS:
                chosen = [r for r in frozen if (r['problem'], r['sd'], r['variant']) == (problem, sd, variant)]
                records.append([problem, sd, variant] + [f'{median(float(r[k]) for r in chosen):.6g}'
                               for k in ('regret', 'top_rmse', 'top_coverage', 'pool_gap')])
    text = table(['Problem', 'SD', 'Variant', 'Regret', 'Top RMSE', 'Top coverage', 'Pool gap'], records)
    path = ROOT / 'validation-outcomes.csv.gz'
    if path.exists():
        outcomes = rows('validation-outcomes.csv.gz')
        records = []
        totals = []
        variants = list(dict.fromkeys(r['policy'] for r in outcomes))
        for problem in ('quadratic2', 'branin2', 'multimodal3', 'ellipsoid6'):
            for sd in ('0.0', '0.03'):
                for variant in variants:
                    chosen = [r for r in outcomes if (r['problem'], r['sd'], r['policy']) == (problem, sd, variant)]
                    records.append([problem, sd, variant] + [f'{median(float(r[k]) for r in chosen):.6g}'
                                   for k in ('best_regret', 'recommendation_regret')])
            baseline = {(r['seed'], r['sd'], r['batch']): r for r in outcomes
                        if (r['problem'], r['policy']) == (problem, 'baseline24')}
            for variant in variants:
                if variant == 'baseline24':
                    continue
                chosen = [r for r in outcomes if (r['problem'], r['policy']) == (problem, variant)]
                record = [problem, variant]
                for metric in ('best_regret', 'recommendation_regret'):
                    delta = [float(r[metric]) - float(baseline[(r['seed'], r['sd'], r['batch'])][metric]) for r in chosen]
                    record.append('/'.join(str(n) for n in (sum(d < -1e-10 for d in delta),
                                  sum(d > 1e-10 for d in delta), sum(abs(d) <= 1e-10 for d in delta))))
                totals.append(record)
        text += '\n\n' + table(['Problem', 'SD', 'Variant', 'Best evaluated', 'Recommendation'], records)
        text += '\n\n' + table(['Problem', 'Variant', 'Best W/L/T', 'Recommendation W/L/T'], totals)
    (ROOT / 'tables.md').write_text(text + '\n')


if __name__ == '__main__':
    summary()
