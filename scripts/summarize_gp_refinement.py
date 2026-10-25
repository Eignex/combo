#!/usr/bin/env python3
"""Validate and archive the GP refinement's paired observations and BO workload measurements."""
import csv
import gzip
import hashlib
import json
import math
import shutil
import statistics as stats
from collections import defaultdict
from pathlib import Path

root = Path('build/reports/gp-refinement')
out = Path('benchmarks/gp-refinement')


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


rows = read(root / 'validation/outcomes.csv')
traces = read(root / 'validation/traces.csv')
protocol = json.loads((out / 'protocol.json').read_text())
assert len(rows) == 864 and len(traces) == 27648
assert {int(r['seed']) for r in rows} == set(protocol['validation_seeds'])
assert all(int(r['calls']) == protocol['external_budget'] for r in rows)
keys = ['problem', 'seed', 'sd', 'batch', 'policy']
groups = defaultdict(list)
for row in traces:
    groups[tuple(row[k] for k in keys)].append(row)
assert len(groups) == len(rows)
for row in rows:
    observations = groups[tuple(row[k] for k in keys)]
    assert [int(r['call']) for r in observations] == list(range(1, 33))
    assert abs(min(float(r['truth']) for r in observations) - float(row['best_regret'])) < 1e-12
    assert float(row['recommendation_regret']) >= 0
    for observation in observations:
        coordinates = list(map(float, observation['coordinates'].split(';')))
        assert len(coordinates) == int(row['problem'][-1]) and all(0 <= x <= 1 for x in coordinates)
for name in protocol['objectives']:
    for seed in protocol['validation_seeds']:
        for sd in ('0.0', '0.03'):
            for batch in ('1', '2', '4'):
                designs = {
                    tuple((r['coordinates'], r['observed']) for r in observations[:8])
                    for key, observations in groups.items() if key[:4] == (name, str(seed), sd, batch)
                }
                assert len(designs) == 1
shutil.copyfile(root / 'validation/outcomes.csv', out / 'outcomes.csv')
(out / 'traces.csv.gz').write_bytes(gzip.compress((root / 'validation/traces.csv').read_bytes(), mtime=0))
for filename in ('factor-cache.csv', 'frozen-scores.csv'):
    shutil.copyfile(root / filename, out / filename)


def median(group, field):
    return stats.median(float(row[field]) for row in group)


def fmt(value):
    return f'{value:.5g}'


def paired(first, second, problem=None, batch=None, sd=None):
    selected = [r for r in rows if (problem is None or r['problem'] == problem)
                and (batch is None or r['batch'] == str(batch)) and (sd is None or r['sd'] == str(sd))]
    baseline = {tuple(r[k] for k in keys[:-1]): float(r['recommendation_regret']) for r in selected if r['policy'] == second}
    differences = [float(r['recommendation_regret']) - baseline[tuple(r[k] for k in keys[:-1])]
                   for r in selected if r['policy'] == first]
    return '/'.join(map(str, (sum(d < -1e-10 for d in differences), sum(d > 1e-10 for d in differences),
                             sum(abs(d) <= 1e-10 for d in differences))))


lines = ['# GP acquisition, search, and posterior-cache refinement', '',
         'This cycle isolates acquisition correctness, sampling/search behavior, and baseline reuse. '
         'It keeps GP kernels, likelihood fitting, sequential batching, and exploration defaults unchanged. '
         'New log-improvement acquisitions and joint local search remain opt-in.', '',
         '## Frozen-posterior diagnosis', '',
         'Twelve frozen cases use the first 24 observations of the archived exclusion/q-NEI histories '
         'for discovery seeds 1501–1503 at noise SD .03. Six q-EI cases have a fixed incumbent; '
         'six q-NEI cases include one additional pending point in the latent baseline. '
         'Each case enumerates six singleton candidates and fifteen pairs.', '',
         'BoTorch receives exactly the same posterior means, covariance, candidate/baseline rows, and '
         '256 antithetic normal draws. All 252 legacy EI values agree within `1e-9`; their mean absolute '
         'differences are around floating-point rounding. Cached EI preserves that agreement, and '
         'the new log acquisitions agree with BoTorch log acquisition within `1e-8` on the same draws. '
         'These checks isolate acquisition and sampling; they do not compare GP fitting or hyperparameter learning.', '',
         'An independent 16,384-draw Sobol reference changes the winning pair in 2/6 q-EI cases '
         'and 0/6 q-NEI cases. Many 256-draw EI estimates are exactly zero because no sampled outcome '
         'improves on the baseline. This is a finite-sample positive-part plateau; it is not proof '
         'of floating-point underflow. No general covariance-calibration or kernel change follows from this audit.', '',
         '| Acquisition | q | Scores | Fixed-normal MAE | High-sample MAE | Zero fraction |',
         '|---|---|---|---|---|---|']
for row in read(out / 'frozen-scores.csv'):
    lines.append('| ' + ' | '.join([row['kind'], row['q'], row['count']] + [fmt(float(row[k])) for k in
                          ('fixed_mae', 'high_sample_mae', 'zero_fraction')]) + ' |')
lines.extend(['', 'Loss aggregation uses Kumulant `MaeLossStat`; zero fractions and expected utilities use '
              'Kumulant summaries. BoTorch versions and raw acquisition values are archived alongside the posterior fixture.', '',
              '## Implementation', '',
              '- `JointLogExpectedImprovement` and `JointLogNoisyExpectedImprovement` evaluate fat-tailed '
              'smoothed improvement directly in log units, using positive-part temperature `1e-6` in original '
              'response units and batch-max temperature `.01` in log-utility units. These are smoothed approximations, '
              'not simply `log(EI)`. Smoothing keeps all-negative sampled improvements ordered.',
              '- `JointLocalAndRandomSearch` starts with pool coordinate exchange, then moves members using '
              'the existing normalized neighborhoods and bounded Klause repair. Every entire candidate set is '
              're-scored, and accepted points retain context, constraints, distinctness, and pending exclusions.',
              '- Joint acquisitions bind completed/pending posterior work, factor its baseline covariance, and '
              'draw its latent outcomes once. Candidate sets use a triangular solve and a conditional covariance '
              'factor of order at most q. The GP also caches training-to-baseline solves, avoiding repeated '
              'baseline-only prediction work. Generic models fall back to full joint prediction.',
              '- Optimized prediction caching is an explicit `JointPosteriorSurrogate` / '
              '`LogJointPosteriorSurrogate` capability. Ordinary delegating decorators use their own joint '
              'overrides, and do not accidentally inherit an underlying model cache that bypasses their transform.',
              '- BO acquisition, GP conditioning, and feasible-search orchestration stay in Combo. '
              'Kumulant retains normal draws, statistical summaries, and Cholesky. Koblas retains matrix products '
              'and triangular operations. Neither dependency gains new utilities or GP-specific APIs.', '',
              '## Baseline-cache workload', '',
              'A standalone JVM workload compares twenty candidate sets per bound round against full joint '
              'prediction/factorization, with identical moments and normals. Binding and baseline factorization '
              'are included; fitting is excluded. Two warmup rounds precede three measured repeats, with execution '
              'order alternating. This measures BO posterior/sampling orchestration, not a replacement for Kumulant’s '
              'factor-kernel benchmark. Times are descriptive measurements on this host, not portable speed guarantees.', '',
              '| Baseline rows | q | Full round ms | Cached round ms | Ratio | Cached utility MAE |',
              '|---|---|---|---|---|---|'])
profile = read(out / 'factor-cache.csv')
for size in (16, 64, 128):
    for q in (1, 4):
        group = [r for r in profile if int(r['baseline']) == size and int(r['q']) == q]
        assert len(group) == 3 and all(int(r['baseline_factorizations']) == 1 and int(r['candidate_factorizations']) == 20 for r in group)
        full, cached = median(group, 'full_nanos') / 1e6, median(group, 'cached_nanos') / 1e6
        lines.append(f'| {size} | {q} | {fmt(full)} | {fmt(cached)} | {fmt(full/cached)} | {fmt(median(group, "mae"))} |')
lines.extend(['', 'The cached path factors the baseline once, then twenty q-by-q conditional matrices. '
              'The full reference factors twenty (baseline+q)-by-(baseline+q) matrices. All compared utilities '
              'agree within `1e-8`; raw per-repeat times and scorer errors are archived. '
              'The 16-row cases were slower with caching in this run; baseline binding and sampling overhead '
              'can outweigh the saved matrix work at small sizes. Larger 64/128-row cases were about 4–7 times faster.', '',
              '## Whole-loop validation', '',
              'Validation uses fresh seeds 1701–1706; development seed 951 is excluded and retained separately. '
              'The fixed protocol covers a weighted quadratic in 2D, normalized Branin in 2D, a multimodal cosine '
              'objective in 3D, and an anisotropic quadratic in 6D. All have a known zero minimum. '
              'Noise SD is 0 or .03, with known variances and unit weights. Batch sizes are 1, 2, and 4. '
              'All 864 runs exhaust 32 external evaluations; the archive contains 27,648 measurements.', '',
              'Every method shares the first eight Latin-hypercube points and their measurements. Later '
              'exploration probability is .2 (1 for random). Model-driven policies share an independent '
              '16-point pool per seed and round. Pool search uses two starts and one exchange pass; local '
              'search adds four sweeps and six attempted neighbors per member, using the same feasible '
              'neighborhood machinery for sequential and joint policies. Joint acquisitions use 256 draws. '
              'All methods use unchanged default GP fitting and standardized responses. No model or search '
              'defaults were selected on validation results.', '',
              'All methods, including random evaluation, use the same GP posterior-mean recommendation '
              'search: 128 independent initial points, four local steps, eight neighbors, and the existing '
              'four-start policy. Thus random means random evaluations plus a GP recommendation. '
              'Results complete in reverse order inside each batch; no pending work carries between rounds. '
              'Dedicated tests cover ongoing pending points, so these runs do not measure asynchronous throughput.', '',
              'Medians below combine the six seeds, two noise levels, and three batch sizes for each objective '
              '(36 outcomes per policy). Smaller regrets are better. Suggestion/recommendation times are '
              'reported separately; the random policy fits its GP during recommendation. External budgets are '
              'equal, computational budgets are not. Times include JVM/shared-host effects; set evaluations '
              'and point evaluations are not equal computational units.', '',
              '| Objective | Policy | Best evaluated regret | Recommendation regret | Suggest ms | Recommend ms |',
              '|---|---|---|---|---|---|'])
for problem in protocol['objectives']:
    for policy in protocol['policies']:
        group = [r for r in rows if r['problem'] == problem and r['policy'] == policy]
        lines.append('| ' + ' | '.join([problem, policy, fmt(median(group, 'best_regret')),
                                      fmt(median(group, 'recommendation_regret')),
                                      fmt(median(group, 'suggestion_nanos') / 1e6),
                                      fmt(median(group, 'recommendation_nanos') / 1e6)]) + ' |')
lines.extend(['', 'Paired recommendation wins/losses/ties for log-NEI local search; ties use absolute '
              'tolerance `1e-10`. Counts are descriptive across correlated settings, not significance tests.', '',
              '| Objective | vs NEI pool | vs log-NEI pool | vs NEI local | vs sequential local |',
              '|---|---|---|---|---|'])
for problem in protocol['objectives']:
    lines.append('| ' + ' | '.join([problem] + [paired('lognei-local', p, problem=problem) for p in
                                              ('nei-pool', 'lognei-pool', 'nei-local', 'sequential-local')]) + ' |')
lines.extend(['', '## Decision and limits', '',
              f'Across all settings, log-NEI local versus NEI pool gives {paired("lognei-local", "nei-pool")} '
              f'wins/losses/ties; versus raw NEI local it gives {paired("lognei-local", "nei-local")}; '
              f'versus sequential local it gives {paired("lognei-local", "sequential-local")}. '
              'Leaving the small pool helps in several settings, but log smoothing does not consistently '
              'outperform raw NEI local search. Sequential selection remains competitive, and the six-dimensional '
              'problem remains difficult. These results do not justify replacing the default batch policy.', '',
              'Baseline reuse is a behavior-preserving computational improvement within numerical tolerance. '
              'Log acquisition and local joint search remain explicit choices. The finite draw count can still '
              'change rankings, and these tests do not integrate GP hyperparameter uncertainty. Baselines are '
              'not pruned; one cubic baseline factor and its storage remain per round. Future work should target '
              'sampling convergence, higher-dimensional posterior diagnostics, and broader tasks before changing '
              'kernels or expanding the surrogate roster.', '',
              '## Reproduction', '',
              'Run `./gradlew refineGp` for the suite and `./gradlew profileGpFactorCache` separately for the '
              'standalone workload. Use `-PgpRefinement.mode=development -PgpRefinement.seeds=951` for the '
              'development smoke run. Run `python3 scripts/summarize_gp_refinement.py` to verify budgets, '
              'pairing, initial designs, coordinates, and measured best regret before archiving.', '',
              'For the independent reference, install the exact versions in `reference.versions.json` '
              '(CPU PyTorch) and run `python scripts/audit_gp_reference.py`. Its default input is the '
              'archived frozen posterior, preserving that comparison independently of later model fitting. '
              '`./gradlew auditGpAcquisition` can regenerate a new posterior export from the old evaluation histories.', '',
              '`outcomes.csv`, `traces.csv.gz`, `frozen.json.gz`, `reference.csv`, `frozen-scores.csv`, and '
              '`factor-cache.csv` contain the complete underlying data. `protocol.json` records fixed settings '
              'and both initial/final source hashes. After extracting the explicit cache capability, all 864 '
              'non-timing outcome rows and all 27,648 evaluation traces replayed exactly; their hashes are retained. '
              'The final full gate `./gradlew check lintDocs --rerun-tasks` passed 268 JVM tests plus API, coverage, '
              'lint, and documentation checks. Numerical and behavioral tests compare cached and full predictions, log transforms, '
              'decorators, baseline-draw reuse, feasibility/context/exclusion, deterministic and singular covariance, '
              'and empty/zero-utility batches.', ''])
(out / 'report.md').write_text('\n'.join(lines))
print(out / 'report.md')
