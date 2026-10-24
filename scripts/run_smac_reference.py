#!/usr/bin/env python3
"""Run the pinned SMAC HPO facade on the positive-cost objectives shared with SmacComparison."""
import argparse
import csv
import math
import os
import time
from importlib.metadata import version
from pathlib import Path
from ConfigSpace import ConfigurationSpace, Float, Integer, Categorical
from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory.dataclasses import TrialValue


def problem(name, seed):
    parameters = [Float('x', (1e-6, 1.0), log=True)] if name == 'logarithmic' else [Float('x', (0.0, 1.0))]
    if name == 'mixed':
        parameters += [Integer('count', (1, 6)), Categorical('mode', ['slow', 'fast'])]
    if name == 'noisy':
        parameters += [Float('y', (0.0, 1.0))]
    space = ConfigurationSpace(seed=seed)
    space.add(parameters)
    return space


def reference(name, config):
    x = config['x']
    if name == 'quadratic':
        return 1 + (x - .73) ** 2
    if name == 'logarithmic':
        return 1 + (math.log(x / .001) / math.log(1000)) ** 2
    if name == 'mixed':
        return 1 + (x - .65) ** 2 + ((config['count'] - 4) / 6) ** 2 + (0 if config['mode'] == 'fast' else .15)
    return 1 + (x - .31) ** 2 + 3 * (config['y'] - .72) ** 2


def noise(config, seed):
    value = math.sin(seed * 12.9898 + config['x'] * 78.233 + config['y'] * 39.425) * 43758.5453
    return .05 * (2 * (value - math.floor(value)) - 1)


def run(name, seed, budget, output):
    space = problem(name, seed)
    scenario = Scenario(space, deterministic=name != 'noisy', n_trials=budget, seed=seed,
                        output_directory=output / 'state' / name / str(seed))
    def target(config, seed=0):
        return reference(name, config) + (noise(config, seed) if name == 'noisy' else 0)
    optimizer = HyperparameterOptimizationFacade(scenario, target, overwrite=True, logging_level=50)
    best = math.inf
    rows = []
    for step in range(budget):
        started = time.perf_counter()
        trial = optimizer.ask()
        truth = reference(name, trial.config)
        cost = target(trial.config, trial.seed or 0)
        optimizer.tell(trial, TrialValue(cost=cost, time=1.0))
        best = min(best, truth)
        incumbent = optimizer.intensifier.get_incumbent()
        recommendation = reference(name, incumbent) if incumbent is not None else None
        rows.append(dict(problem=name, algorithm='smac-2.4.1', seed=seed, evaluations=step + 1,
                         cumulative_cost=step + 1, observed_response=cost, best_reference=best, simple_regret=max(0, best - 1),
                         x=trial.config['x'], y=trial.config.get('y', ''), count=trial.config.get('count', ''),
                         mode=trial.config.get('mode', ''), eval_seed=trial.seed,
                         recommendation_regret=max(0, recommendation - 1) if recommendation is not None else "",
                         duration_seconds=time.perf_counter() - started))
    path = output / f'smac-{name}-{seed}.csv'
    with path.open('w') as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f'{name} seed={seed}: regret={best - 1:.6g}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('build/reports/smac-comparison'))
    parser.add_argument('--budget', type=int, default=32)
    parser.add_argument('--seeds', type=int, nargs='+', default=[3, 11, 19, 29, 43, 61])
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("Run with PYTHONHASHSEED=0 for reproducible ConfigSpace hashing")
    if version('smac') != '2.4.1':
        raise RuntimeError('Install benchmarks/smac-requirements.txt; this comparison requires SMAC 2.4.1')
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ('quadratic', 'logarithmic', 'mixed', 'noisy'):
        for seed in args.seeds:
            run(name, seed, args.budget, args.output)
