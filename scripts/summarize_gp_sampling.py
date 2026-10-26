#!/usr/bin/env python3
"""Summarize Kumulant-scored sampling errors and paired BO outcomes without refitting models."""
import csv
import gzip
import json
from pathlib import Path
from statistics import mean, median

ROOT = Path('benchmarks/gp-qmc')


def rows(name):
    path = ROOT / name
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as stream:
        return list(csv.DictReader(stream))


def table(headers, records):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, record)) + ' |' for record in records])


def main():
    frozen = {method: rows(method + '-convergence-scores.csv.gz') for method in ('antithetic', 'scrambledsobol')}
    scores = []
    for kind in ('raw', 'log'):
        for q in ('1', '2', '4'):
            for n in ('256', '1024', '4096'):
                record = [kind, q, n]
                for method in frozen:
                    selected = [r for r in frozen[method] if (r['kind'], r['q'], r['samples']) == (kind, q, n)]
                    threshold = 1e-6 if kind == 'raw' else -13.815510557964274
                    active = [r for r in selected if float(r['reference_best']) > threshold]
                    record.extend([f'{mean(float(r["within_one_percent"]) for r in active):.1%}',
                                   f'{mean(float(r["relative_regret"]) for r in active):.2%}'])
                scores.append(record)
    outcomes = rows('validation-outcomes.csv.gz')
    paired = []
    wins = []
    for problem in ('quadratic2', 'branin2', 'multimodal3', 'ellipsoid6'):
        for q in ('1', '4'):
            for sd in ('0.0', '0.03'):
                selected = [r for r in outcomes if (r['problem'], r['batch'], r['sd']) == (problem, q, sd)]
                anti = {r['seed']: r for r in selected if r['policy'] == 'lognei-local'}
                sobol = {r['seed']: r for r in selected if r['policy'] == 'lognei-local-sobol'}
                record = [problem, q, sd]
                for metric in ('best_regret', 'recommendation_regret'):
                    record.extend([f'{median(float(r[metric]) for r in anti.values()):.6g}',
                                   f'{median(float(r[metric]) for r in sobol.values()):.6g}'])
                    differences = [float(sobol[seed][metric]) - float(anti[seed][metric]) for seed in anti]
                    wins.append((problem, metric, differences))
                paired.append(record)
    totals = []
    for problem in ('quadratic2', 'branin2', 'multimodal3', 'ellipsoid6'):
        record = [problem]
        for metric in ('best_regret', 'recommendation_regret'):
            differences = [delta for name, key, values in wins if (name, key) == (problem, metric) for delta in values]
            record.append('/'.join(map(str, [sum(d < -1e-10 for d in differences),
                                            sum(d > 1e-10 for d in differences),
                                            sum(abs(d) <= 1e-10 for d in differences)])))
        totals.append(record)
    throughput = json.loads((ROOT / 'normal-throughput.json').read_text())
    timings = []
    for result in throughput:
        p = result['params']
        m = result['primaryMetric']
        timings.append([result['benchmark'].rsplit('.', 1)[-1], p['dimensions'], p['sampling'],
                        f'{m["score"]:.1f} ± {m["scoreError"]:.1f}', m['scoreUnit']])
    quality = [[r['dimensions'], r['samples'], r['sampling'], f'{float(r["rmse"]):.6g}']
               for r in rows('normal-quality.csv')]
    text = '''# Scrambled Sobol sampling refinement

Kumulant supplies a general SobolSequence; Combo adds opt-in base-sample selection for qEI,
qNEI, qLogEI, and qLogNEI. Existing constructors and antithetic defaults remain compatible.
The results do not support changing the 256-draw default: frozen ranking gains are modest and
budget-dependent, and fresh whole-loop regret varies across objectives and noise conditions.
No Koblas implementation changes are required.

## Numerical machinery and ownership

Kumulant generates uniform points with the first 1,024
[Joe-Kuo D(6) directions](https://web.maths.unsw.edu.au/~fkuo/sobol/) and LMS plus digital-shift
scrambling, a standard Sobol randomization described by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html).
Normal points use AS 241's inverse CDF on 52-bit cell midpoints. This keeps endpoints finite without
dropping initial points or clipping a continuous normal tail onto an endpoint. Direction-table and
quantile provenance, notices, independent reference tests, and a table regeneration script live in Kumulant.

A seeded engine consumes a fixed randomization stream per coordinate. Coordinate projections and
sample prefixes agree as dimensions and draw counts increase. Combo uses these projections when
adding conditional candidate coordinates, retaining cached baseline draws and factorization work.
Acquisition definitions, pending exclusion, feasible search, and the suggestion lifecycle stay in Combo.
Matrix products and triangular operations continue using Koblas; losses and summaries use Kumulant.

Sobol points are dependent integration points, not IID normal observations. Balanced draw counts
are powers of two. Combo's opt-in policy supports at most 1,024 latent coordinates, including active
baseline and candidate coordinates; this is unrelated to the number of configuration features.
An explicit error directs larger histories to Antithetic; there is no silent fallback or history truncation.

## Frozen acquisition ranking

Reuse all 20 posteriors from the previous follow-up: 12 earlier qEI/qNEI histories, including pending
points, plus eight noisy histories spanning four objectives and two, three, and six feature dimensions.
Each has six candidate coordinates: six singletons, 15 pairs, and 15 quartets. Hold all posterior means,
covariances, baselines, candidate sets, and smoothing parameters fixed. Compare both samplers with
eight streams at 256/1,024/4,096 draws. Kumulant's score stats measure errors against each of the
three archived independent 65,536-draw BoTorch Sobol scrambles.

The table averages assessments across those three references. These repeated assessments are
correlated, not 480 independent posterior experiments. Active cases have reference best utility
above 1e-6; the archive also retains tiny-utility tails and all absolute acquisition errors.
“Within 1%” means selected-set reference utility is within 1% of the reference best over the same
finite candidate sets. Relative regret is 1 - selected/best (using log differences for log acquisition).
References are Monte Carlo estimates and can disagree, especially on rare improvements or near ties.

''' + table(['Kind', 'q', 'Draws', 'Anti within 1%', 'Anti mean relative regret',
             'Sobol within 1%', 'Sobol mean relative regret'], scores) + '''

At q=4 and 4,096 draws the log-acquisition comparison improves modestly across the three references.
At 256 draws there is no consistent advantage in both ranking metrics. This is evidence about a
fixed finite pool; it is not a convergence guarantee or proof of better optimization trajectories.

The independent captured-normal BoTorch audit adds 432 Sobol candidate sets across q=1/2/4.
Raw values agree within 1e-9 and log values within 1e-8. JVM tests cover both the previous antithetic
fixtures and these Sobol fixtures, plus batch-size growth, reordering, exclusions, and cache reuse.

## Fresh whole-loop validation

Use previously unused seeds 1801–1806 with four objectives, SD 0/.03, q=1/4, and two samplers:
192 runs and 6,144 objective evaluations. Each run has the same 32-evaluation budget, initial design,
exploration probability, 16-point initialization pools, bounded feasible local search, GP fitting
budget, known measurement variances, standardization, and independent recommendation-search budget
as the prior refinement suite. Both methods use qLogNEI with 256 draws. Completions reverse within
each batch; no pending evaluations carry between rounds. Only the normal base-sample policy varies.
Truth scores outcomes and never enters fitting, acquisition scoring, or recommendation selection.
These results were not used to tune sampling parameters or choose a new default.

Each table cell is the median of six paired seeds; smaller regret is better.

''' + table(['Problem', 'q', 'SD', 'Anti best regret', 'Sobol best regret',
             'Anti recommendation regret', 'Sobol recommendation regret'], paired) + '''

Paired Sobol wins/losses/ties versus antithetic (absolute tie tolerance 1e-10), 24 pairs per objective:

''' + table(['Problem', 'Best evaluated', 'Recommendation'], totals) + '''

There is no uniform end-to-end advantage, so sampling remains opt-in. These runs do not isolate
model error, acquisition-search error, and Monte Carlo error. The prior
[fixed-history recommendation diagnostics](../gp-followups/report.md) treated model and search
separately; a sampling comparison alone does not identify the cause of remaining recommendation errors.

## General normal integration quality and cost

Kumulant's quality runner estimates E[cos(0.2 sum(X))] = exp(-0.02 d) for X ~ N(0, I_d), using
eight independent seeds at each dimension and sample count. Kumulant MeanStat and MseLossStat
compute estimates and RMSE. The pseudo-random reference here is IID, not antithetic. Sobol helps
this integrand in lower dimensions, but the high-dimensional results are mixed.

''' + table(['Dimensions', 'Draws', 'Sampler', 'RMSE'], quality) + '''

Normal-generation JMH results below use two forks, three one-second warmups, five one-second
measurements per fork, and 256 points per invocation. `draw` reuses an engine and output storage;
`bindAndDraw` also includes a newly seeded engine. Throughput is blocks/second, not points/second.
JMH reports its score error alongside the estimate. The shared Intel i9-12900H host had concurrent
workloads; the large uncertainty on some cells precludes precise general timing claims.
These engine timings exclude GP fitting, factorization, posterior prediction, and acquisition scoring.

''' + table(['Operation', 'Dimensions', 'Sampler', 'JMH score ± error', 'Unit'], timings) + '''

## Using the policy

```kotlin
val acquisition = JointLogNoisyExpectedImprovement(
    sampling = JointNormalSampling.ScrambledSobol,
    samples = 1024,
)
val policy = JointBatchPolicy(acquisition, jointSearch)
```

Existing `JointLogNoisyExpectedImprovement()` retains Antithetic and 256 draws. The same explicit
sampling constructor is available on the other three joint improvement factories. A bound objective
is single-threaded and keeps its normal base samples fixed across candidate scoring.

## Reproduction

```sh
./gradlew refineGpSampling -Pqmc.mode=frozen
./gradlew refineGpSampling -Pqmc.mode=reference
python scripts/audit_gp_reference.py --input build/reports/gp-qmc/frozen.json \\
  --output build/reports/gp-qmc/reference.csv
./gradlew refineGpSampling -Pqmc.mode=loop
./gradlew check lintDocs
python scripts/summarize_gp_sampling.py
```

During cross-library development add `-Pkumulant.local` with the Kumulant checkout adjacent to Combo.
The final dependency pins consume the published Kumulant sampler; a sibling checkout is not required.
Compressed CSVs preserve both frozen methods, their Kumulant-scored errors, and every fresh outcome
and trace. The posterior inputs and reference scrambles remain in benchmarks/gp-followups.
Metadata records tested revisions, seeds, and budgets. Copy new outputs into this directory before
regenerating the report. The captured-normal fixtures and independent reference are retained here.

In Kumulant, use the existing benchmark engine:

```sh
./kumulant-bench/benchmark.sh --quality --output /tmp/normal-quality
./kumulant-bench/benchmark.sh --output /tmp/normal-throughput -- \\
  '.*NormalSamplingBenchmark.*' -p samples=256 -p dimensions=32,128 \\
  -p sampling=pseudo,sobol -f 2 -wi 3 -i 5 -w 1s -r 1s
```
'''
    (ROOT / 'report.md').write_text(text)


if __name__ == '__main__':
    main()
