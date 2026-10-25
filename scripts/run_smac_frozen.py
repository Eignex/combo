#!/usr/bin/env python3
"""Export pinned SMAC predictions on refineBo histories; score them with the scoreSmacFrozen task."""
import csv
import os
from importlib.metadata import version
from pathlib import Path
import numpy as np
from ConfigSpace import ConfigurationSpace, Float
from smac.model.random_forest.random_forest import RandomForest

if version('smac') != '2.4.1' or os.environ.get('PYTHONHASHSEED') != '0':
    raise RuntimeError('Use the locked SMAC environment with PYTHONHASHSEED=0')
root = Path('build/reports/bo-refinement')
with (root / 'smac-predictions.csv').open('w') as destination:
    writer = csv.writer(destination, lineterminator='\n')
    writer.writerow(['problem', 'seed', 'history', 'prediction', 'variance', 'truth'])
    for path in sorted(root.glob('frozen-*.csv')):
        _, problem, seed, size = path.stem.split('-')
        if int(seed) not in range(201, 221):
            continue
        rows = list(csv.DictReader(path.open()))
        columns = [c for c in rows[0] if c.startswith('x')]
        training = [r for r in rows if r['role'] == 'train']
        queries = [r for r in rows if r['role'] == 'query']
        x = np.array([[float(r[c]) for c in columns] for r in training])
        y = np.array([float(r['response']) for r in training])
        q = np.array([[float(r[c]) for c in columns] for r in queries])
        truth = np.array([float(r['truth']) for r in queries])
        space = ConfigurationSpace(seed=int(seed))
        space.add([Float(c, (0.0, 1.0)) for c in columns])
        model = RandomForest(space, n_trees=10, ratio_features=1.0, min_samples_split=2,
                             min_samples_leaf=1, log_y=False, seed=int(seed))
        model.train(x, y)
        means, variances = model.predict(q)
        for prediction, variance, target in zip(means.ravel(), variances.ravel(), truth):
            writer.writerow([problem, seed, size, prediction, variance, target])
        print(path.stem, flush=True)
