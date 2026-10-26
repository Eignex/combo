#!/usr/bin/env python3
"""Summarize archived diagnostics; losses and coverage are scored with Kumulant in the JVM harness."""
import csv
import gzip
from pathlib import Path
from statistics import median, mean

ROOT = Path('benchmarks/gp-followups')


def rows(name):
    path = ROOT / name
    stream = gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()
    with stream:
        return list(csv.DictReader(stream))


def table(headers, records):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, record)) + ' |' for record in records])


def med(data, field):
    return median(float(row[field]) for row in data)


def main():
    recommendations = rows('recommendations.csv.gz')
    acquisition = rows('convergence-scores.csv.gz')
    allocation = []
    cache = []
    for fork in range(1, 4):
        for row in rows(f'allocation-paired-{fork}.csv'):
            row['fork'] = str(fork)
            allocation.append(row)
        cache.extend(rows(f'factor-cache-{fork}.csv'))
    rec_table = []
    for problem in ('quadratic2', 'ellipsoid6'):
        for sd in ('0.0', '0.03'):
            base = [r for r in recommendations if r['problem'] == problem and r['sd'] == sd]
            for trials, search in [('24', 'standard'), ('24', 'large'), ('96', 'standard'), ('96', 'large')]:
                selected = [r for r in base if r['trials'] == trials and r['search'] == search]
                rec_table.append([problem, sd, trials, search, f'{med(selected, "regret"):.6g}',
                                  f'{med(selected, "finite_pool_gap"):.3g}', f'{med(selected, "top_rmse"):.3g}'])
    sampling_table = []
    for kind in ('raw', 'log'):
        for q in ('1', '2', '4'):
            for count in ('256', '1024', '4096'):
                selected = [r for r in acquisition if (r['kind'], r['q'], r['samples'], r['reference']) ==
                            (kind, q, count, 'sobol-0')]
                # Keep tiny-utility cases visible in the full column; separate them from actionable ranking differences.
                threshold = 1e-6 if kind == 'raw' else -13.815510557964274
                active = [r for r in selected if float(r['reference_best']) > threshold]
                sampling_table.append([kind, q, count, f'{mean(float(r["within_one_percent"]) for r in selected):.1%}',
                                       f'{mean(float(r["within_one_percent"]) for r in active):.1%}',
                                       f'{mean(float(r["relative_regret"]) for r in active):.2%}'])
    perf_table = []
    for size in ('16', '64', '128'):
        for q in ('1', '4'):
            for kind in ('raw', 'log'):
                selected = [r for r in allocation if (r['baseline'], r['q'], r['kind']) == (size, q, kind)]
                current = [r for r in selected if r['implementation'] == 'current']
                reference = {(r['fork'], r['repeat']): r for r in selected if r['implementation'] == 'reference'}
                ratios = [int(reference[r['fork'], r['repeat']]['nanos']) / int(r['nanos']) for r in current]
                before = median(int(r['bytes']) / int(r['evaluations']) for r in reference.values())
                after = median(int(r['bytes']) / int(r['evaluations']) for r in current)
                perf_table.append([size, q, kind, f'{before:.0f}', f'{after:.0f}', f'{1-after/before:.1%}',
                                   f'{median(ratios):.2f}×'])
    cache_table = []
    for size in ('16', '64', '128'):
        for q in ('1', '4'):
            selected = [r for r in cache if (r['baseline'], r['q']) == (size, q)]
            cache_table.append([size, q, f'{med(selected, "full_nanos")/1e6:.3f}',
                                f'{med(selected, "cached_nanos")/1e6:.3f}',
                                f'{median(int(r["full_nanos"])/int(r["cached_nanos"]) for r in selected):.2f}×'])
    stability = []
    for kind in ('raw', 'log'):
        for q in ('1', '2', '4'):
            primary = [r for r in acquisition if (r['kind'], r['q'], r['samples'], r['replicate']) ==
                       (kind, q, '65536', '8') and r['reference'].startswith('sobol-')]
            stability.append([kind, q, f'{mean(float(r["within_one_percent"]) for r in primary):.1%}'])
    text = '''# GP recommendation, sampling, and allocation follow-ups

The recommendation gap has different causes in these fixtures. More search helps the noiseless
quadratic, but its noisy histories already have small finite-pool posterior-mean search gaps.
A larger GP fitting budget helps the noiseless ellipsoid and can worsen its noisy recommendations.
Joint sampling still has consequential finite-sample ranking errors, especially at q=4.
No recommendation, GP fitting, acquisition, or Monte Carlo defaults change in this refinement.

The production change reuses acquisition scratch buffers and the Kumulant mean accumulator,
precomputes fixed smoothing terms, and uses the exact single-point smooth-maximum identity.
Acquisition definitions remain in Combo. Kumulant supplies losses, summaries, normal sampling,
and Cholesky; Koblas supplies matrix products and triangular operations. Neither dependency changes.

## Frozen recommendation diagnosis

These are discovery diagnostics on the already published 1701–1706 histories, not new validation.
Take all 216 final 32-evaluation histories for quadratic2/ellipsoid6, noise SD 0/.03, q=1/2/4,
and sequential-local/nei-local/lognei-local. Restore reverse completion order within each batch.
The 24-trial standard-search reconstruction matches all original recommendations within 1e-10.
Responses, known observation variances, unit weights, and standardization match the original run.
Truth is used only for evaluation; neither fitting nor recommendation search sees it.

For each fixed history compare GP hyperparameter budgets 24 and 96, and recommendation search:
standard = 128 random candidates, 4 starts, 4 steps, 8 neighbors;
large = 1,024 candidates, 4 starts, 16 steps, 16 neighbors.
Both include evaluated configurations as anchors. Each fit also predicts a shared independent
4,096-point uniform set. The finite-pool gap is recommendation mean minus the minimum mean over
that set, history, and both recommendations; it is not a continuous-global-optimum certificate.
Top RMSE uses the best true decile of that independent set. Kumulant MseLossStat scores errors;
MeanStat scores diagnostic two-standard-deviation latent coverage. CSVs retain likelihood,
length scales, coverage, best true evaluated point, and model-best evaluated point.

Each table cell aggregates 54 histories (six seeds, three batch sizes, three policies).

''' + table(['Problem', 'SD', 'Fit trials', 'Search', 'Median true regret', 'Median pool gap', 'Top RMSE'], rec_table) + '''

Increasing search can improve the model score while worsening true regret under noise.
The latent-best evaluated point is an oracle diagnostic, not an implementable recommendation policy;
its advantage does not establish a bug in posterior-mean recommendation. More marginal-likelihood
search is also not uniformly better. These results support retaining configurable independent
recommendation budgets and testing model/prior changes separately on fresh seeds.

## Monte Carlo convergence

Freeze 20 posteriors: the 12 earlier qEI/qNEI fixtures (including pending points in qNEI), plus
24-observation noisy lognei-local histories for four objectives, seeds 1701/1702, q=4.
The latter include two-, three-, and six-dimensional fits. Each posterior has six candidate points;
score all six singletons, 15 pairs, and 15 quartets. Fits, points, and baselines never change.

Use eight antithetic pseudo-random streams at 256, 1,024, and 4,096 draws, plus two independent
65,536-draw streams. These streams are not nested across draw counts. Compare with three independent
65,536-draw BoTorch Sobol scrambles, no baseline pruning, no root caching, double precision.
BoTorch versions are archived. Monte Carlo references are estimates, not exact ground truth.
Kumulant MaeLossStat records acquisition error against each reference; the archived score table also
records selected-set utility, relative regret, ties, and zero fractions.

The following uses the first Sobol scramble (160 selections per row: 20 posteriors × eight streams).
“Within 1%” compares the selected set's reference utility with the best of these same finite sets.
The active subset excludes reference best utility ≤1e-6, explicitly separating tiny-utility tails
from ranking errors of larger magnitude. It has 16 posteriors / 128 selections per row.
Relative regret is 1 − selected/best, using exp(log-selected − log-best) for log acquisition.

''' + table(['Kind', 'q', 'Draws', 'All within 1%', 'Active within 1%', 'Active mean relative regret'], sampling_table) + '''

Even the large pseudo-random reference is not fully stable. The table below compares the first
65,536-draw antithetic stream against all three Sobol scrambles (60 selections per row).
Rare improvements and near ties can change winners; a relative error near one on utility around
1e-18 has a different practical meaning from the same relative error on a useful acquisition.
These diagnostics do not justify claiming q=4 convergence at 4,096 or changing defaults based solely
on a finite-pool experiment. A variance-reducing sampler and acquisition-ranking stability are
better follow-ups than a blanket draw-count increase; general sampling primitives belong in Kumulant.

''' + table(['Kind', 'q', '65,536-draw selections within 1%'], stability) + '''

The independent captured-normal audit now covers q=4 as well: 432 candidate sets across 12 frozen
posteriors, for both raw and log acquisitions, agree with BoTorch within 1e-9 (raw) and 1e-8 (log). These fixtures are checked in JVM tests,
including alternating batch sizes so reused scratch state cannot leak between evaluations.

## Allocation and timing

JFR allocation samples identified per-draw smoothing arrays and mean accumulators as a large source
of temporary objects. The optimized bound objective owns reusable score/improvement storage and
resets its Kumulant accumulator. For q=1 the smooth maximum is exactly its sole log improvement.
Existing common-normal assignment, reduction order, and single-threaded bound-objective contract remain.

The paired benchmark retains the pre-change acquisition implementation from 53ead84 as a test-only
control. Each independent JVM alternates implementation and raw/log order, uses 500 whole-round
warmups, then measures 30 rounds. Three JVMs give 90 pairs per cell. A round evaluates 20 candidate
sets at 256 draws; fitting and baseline binding are excluded here. Checksums agree within 1e-12.
ThreadMXBean measures current-thread allocation bytes; JFR recordings are separate from timing runs.
The host is an Intel i9-12900H, Temurin 25.0.1, with concurrent workloads. Timings are diagnostic;
allocation reductions are the more reproducible result. No wall-clock regression gate is introduced.

''' + table(['Baseline', 'q', 'Kind', 'Reference B/eval', 'Current B/eval', 'Reduction', 'Paired median speedup'], perf_table) + '''

Repeat the full-versus-cached posterior-factor benchmark with 30 warmup rounds, 15 measured rounds,
and three independent JVMs, alternating evaluation order. Each round scores 20 sets at 256 draws;
fitting is excluded and baseline binding included. The same GP moments and antithetic normals feed
both paths; Kumulant MaeLossStat verifies utility agreement below 1e-8. Cached factor orders are
B once per round and q per set, versus B+q per set in the full reference.

''' + table(['Baseline', 'q', 'Full median ms/round', 'Cached median ms/round', 'Paired median speedup'], cache_table) + '''

Longer warmup removes the previous apparent small-baseline cache regression in these measurements.
There is no evidence here for adding a special small-baseline cutoff.

## Reproduction

Run from the repository root; these optional experiments are not part of the normal test gate:

```sh
./gradlew auditGpFollowups -Pfollowup.mode=recommendation
./gradlew auditGpFollowups -Pfollowup.mode=convergence
# Optional external reference environment: torch, botorch, gpytorch, numpy, scipy.
python scripts/audit_gp_convergence.py
./gradlew auditGpFollowups -Pfollowup.mode=score
for fork in 1 2 3; do
  ./gradlew auditGpFollowups -Pfollowup.mode=allocation -Pfollowup.fork=paired-$fork
  ./gradlew profileGpFactorCache -Pfollowup.fork=$fork
done
./gradlew auditGpAcquisition
python scripts/audit_gp_reference.py --input build/reports/gp-refinement/frozen.json \\
  --output build/reports/gp-followups/reference-q4.csv
./gradlew check lintDocs
python scripts/summarize_gp_followups.py
```

Results are written under build/reports/gp-followups and build/reports/gp-refinement.
The checked-in compressed CSVs preserve the measured run; copy new measurements into this directory
before regenerating this report. JFR can be enabled separately with
`-Pfollowup.jfr=/absolute/path/recording.jfr`; its sampled stack summary from the final paired
benchmark is archived in allocation-sites.csv. We group allocation samples by their first Combo
stack frame; sampled weights estimate allocation volume, not retained heap. Use `jfr print --json
--events jdk.ObjectAllocationSample /absolute/path/recording.jfr` to inspect the raw recording.
'''
    (ROOT / 'report.md').write_text(text)


if __name__ == '__main__':
    main()
