#!/usr/bin/env python3
"""Archive the predeclared paired joint-batch experiment and regenerate its report."""
import csv
import gzip
import hashlib
import json
import shutil
import statistics as stats
import subprocess
from collections import defaultdict
from pathlib import Path

source = Path('build/reports/joint-batch')
out = Path('benchmarks/joint-batch')
out.mkdir(parents=True, exist_ok=True)


def read(name):
    with (source / name).open() as stream:
        return list(csv.DictReader(stream))


outcomes = read('outcomes.csv')
traces = read('traces.csv')
assert len(outcomes) == 80 and len(traces) == 2560
assert {int(r['seed']) for r in outcomes} == set(range(1501, 1509))
assert all(int(r['calls']) == 32 for r in outcomes)
groups = defaultdict(list)
for row in traces:
    groups[(row['seed'], row['sd'], row['algorithm'])].append(row)
assert len(groups) == len(outcomes)
for rows in groups.values():
    assert [int(r['call']) for r in rows] == list(range(1, 33))
for seed in range(1501, 1509):
    for sd in ('0.0', '0.03'):
        designs = {
            tuple((r['x'], r['y'], r['observed']) for r in rows[:8])
            for (s, noise, _), rows in groups.items() if s == str(seed) and noise == sd
        }
        assert len(designs) == 1, 'initial design and measurement keys must match across methods'
shutil.copyfile(source / 'outcomes.csv', out / 'outcomes.csv')
(out / 'traces.csv.gz').write_bytes(gzip.compress((source / 'traces.csv').read_bytes(), mtime=0))


def median(rows, field):
    return stats.median(float(row[field]) for row in rows)


def fmt(value):
    return f'{value:.5g}'


lines = ['# Joint batch acquisition', '',
         'A paired q=2 experiment compares joint acquisition with existing batch policies. '
         'Validation seeds 1501–1508 were reserved before the development run (seed 941); '
         'the development results are archived separately and excluded below. '
         'No defaults were tuned on these results. Joint policies remain opt-in.', '',
         '## Protocol', '',
         '- Objective: `1 + (x - .31)^2 + 3(y - .72)^2` on the unit square, minimized. '
         'Noise standard deviation is 0 or .03, with known per-observation variances and unit weights. '
         'Kumulant supplies reproducible normal measurement noise.',
         '- Every run has 32 external measurements in 16 batches of two. The initial eight '
         'Latin-hypercube points and their measurements match exactly across methods. '
         'Subsequent exploration probability is .2 (100% for random sampling). '
         'Completion arrives in reverse order within each batch; no pending evaluations remain between rounds.',
         '- Every model uses the unchanged default GP and standardized responses. All five sampling methods '
         'use the same posterior-mean GP recommendation search with 64 initial candidates and feasible local search. '
         'Thus the random row means random evaluations plus a GP recommendation, not best noisy observed response.',
         '- Model-driven policies share a 16-point independent random candidate pool per seed and round, '
         'unaffected by acquisition random consumption. Sequential exclusion and posterior-mean fantasies '
         'select from that pool with local search disabled. Joint search uses two starts and one coordinate-exchange '
         'pass with 256 antithetic Monte Carlo samples. Assignments are distinct and completed points are excluded. '
         'Study validates every proposed assignment through Klause.',
         '- q-EI uses the observed best response when deterministic and the best posterior mean at evaluated '
         'coordinates when noisy. q-NEI integrates the latent baseline jointly with candidates. '
         'Neither uses true objective values in fitting, search, or recommendation.',
         '- Recommendation regret and best evaluated latent regret measure different decisions. '
         'The search is finite-pool and the objective is one smooth two-dimensional example. '
         'Eight seeds do not establish broad optimizer rankings. No wall-time speedup is claimed: '
         'a whole-set acquisition evaluation is more expensive than a single-point score.', '',
         '## Validation results', '',
         'Medians across eight paired seeds; smaller regrets are better. Fits include the final recommendation. '
         'Mean-fantasy fits include existing temporary model fits. Acquisition counts measure points for '
         'sequential policies and complete sets for joint policies.', '',
         '| Noise SD | Policy | Best evaluated regret | Recommendation regret | Fits | Acquisition evaluations |',
         '|---|---|---|---|---|---|']
for sd in (0.0, 0.03):
    for method in ('random', 'exclusion', 'mean-fantasy', 'qei', 'qnei'):
        rows = [r for r in outcomes if float(r['sd']) == sd and r['algorithm'] == method]
        lines.append('| ' + ' | '.join([str(sd), method] + [fmt(median(rows, field)) for field in (
            'best_regret', 'recommendation_regret', 'model_fits', 'acquisition_evaluations')]) + ' |')
lines.extend(['', 'Paired recommendation wins/losses/ties against sequential exclusion (strict numerical comparison):', '',
              '| Noise SD | Policy | Wins | Losses | Ties | Median paired regret difference |',
              '|---|---|---|---|---|---|'])
for sd in (0.0, 0.03):
    baseline = {r['seed']: float(r['recommendation_regret']) for r in outcomes
                if float(r['sd']) == sd and r['algorithm'] == 'exclusion'}
    for method in ('mean-fantasy', 'qei', 'qnei'):
        differences = [float(r['recommendation_regret']) - baseline[r['seed']] for r in outcomes
                       if float(r['sd']) == sd and r['algorithm'] == method]
        lines.append(f'| {sd} | {method} | {sum(d < 0 for d in differences)} | '
                     f'{sum(d > 0 for d in differences)} | {sum(d == 0 for d in differences)} | '
                     f'{fmt(stats.median(differences))} |')
lines.extend(['', 'Joint acquisition did not improve recommendation regret over sequential exclusion here. '
              'Under noise, q-EI won 1/8 paired comparisons and lost 6; q-NEI won 2 and lost 5. '
              'The finite pool often gives all model-driven policies the same best evaluated point, '
              'while their other measurements change the posterior recommendation. '
              'These results support keeping the policies opt-in, not changing the default batch strategy.', '',
              '## Implementation and limits', '',
              'Analytic-limit tests verify q=1 normal EI, two-point positive and negative correlation, '
              'uncertain-baseline EI, minimization, log-response sampling, and zero covariance. '
              'Integration tests cover pending exclusion, context-scoped baselines, frozen-model reuse, '
              'state transfer, out-of-order completion, feasibility, partial exhaustion, and atomic reservation failures.', '',
              'Joint acquisition reuses Kumulant normal sampling, means, and Cholesky, plus Koblas matrix products. '
              'Numerical diagonal stabilization is bounded and distinct from observation noise. '
              'Exactly duplicate coordinates share one draw and deterministic coordinates receive no artificial noise.', '',
              'The initial implementation factors each queried joint covariance. No baseline pruning or cached root '
              'is implemented, so q-NEI scales poorly to large histories. Finite Monte Carlo EI can have zero-improvement '
              'plateaus; log-improvement methods and larger benchmark suites are useful follow-ups. '
              'Pending behavior is additionally tested directly; these synchronous runs do not measure asynchronous throughput.', '',
              '## Reproduction and data', '',
              'Run `./gradlew compareJointBatch` and `python3 scripts/summarize_joint_batch.py`. '
              'Use `-Pjoint.seeds=941` only for the separate development run. '
              '`outcomes.csv` contains all 80 validation outcomes, and `traces.csv.gz` contains all 2,560 evaluations '
              'with measurement, latent truth, coordinates, and batch. `protocol.json` records source hashes '
              'and the base commit. Development outcomes and traces remain under `development/`.', ''])
(out / 'report.md').write_text('\n'.join(lines))
paths = [Path('build.gradle.kts')] + sorted(Path('src/jvmMain/kotlin/com/eignex/combo/bo').rglob('*.kt')) + [
    Path('src/jvmTest/kotlin/com/eignex/combo/bo/JointBatchBenchmark.kt'),
    Path('src/jvmTest/kotlin/com/eignex/combo/bo/NoisyBoBenchmark.kt'),
    Path('src/jvmTest/kotlin/com/eignex/combo/bo/RefinementBenchmark.kt'),
]
metadata = {
    'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'validation_seeds': list(range(1501, 1509)), 'development_seeds': [941],
    'budget': 32, 'batch_size': 2, 'noise_sd': [0.0, 0.03],
    'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
}
(out / 'protocol.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(out / 'report.md')
