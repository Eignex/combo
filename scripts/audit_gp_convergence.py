#!/usr/bin/env python3
"""Optional independent Sobol reference for the frozen Combo convergence experiment."""
import argparse
import csv
import gzip
import itertools
import json
import warnings
from importlib.metadata import version
from pathlib import Path

import gpytorch
import torch
from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from audit_gp_reference import FrozenPosterior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('build/reports/gp-followups/posteriors.json'))
    parser.add_argument('--output', type=Path, default=Path('build/reports/gp-followups/sobol.csv'))
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    warnings.filterwarnings('ignore', category=Warning, module='botorch')
    payload = gzip.decompress(args.input.read_bytes()).decode() if args.input.suffix == '.gz' else args.input.read_text()
    rows = []
    for fixture in json.loads(payload):
        model = FrozenPosterior(fixture)
        baseline = fixture['baseline']
        noisy = fixture['id'].endswith('qnei')
        for replicate in range(3):
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([65536]), seed=9123 + replicate)
            common = dict(model=model, sampler=sampler)
            if noisy:
                common.update(X_baseline=torch.arange(baseline).double().unsqueeze(-1),
                              prune_baseline=False, cache_root=False)
                raw = qNoisyExpectedImprovement(**common)
                log = qLogNoisyExpectedImprovement(**common)
            else:
                common['best_f'] = fixture['incumbent']
                raw, log = qExpectedImprovement(**common), qLogExpectedImprovement(**common)
            for q in (1, 2, 4):
                for indices in itertools.combinations(range(baseline, len(fixture['means'])), q):
                    x = torch.tensor(indices).double().reshape(1, q, 1)
                    with torch.no_grad(), gpytorch.settings.fast_computations(covar_root_decomposition=False):
                        rows.append(dict(fixture=fixture['id'], q=q, replicate=replicate, set=';'.join(map(str, indices)),
                                         raw=raw(x).item(), log=log(x).item()))
        print(f"Sobol reference {fixture['id']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix('.versions.json').write_text(json.dumps({
        name: version(name) for name in ('torch', 'botorch', 'gpytorch', 'numpy', 'scipy')}, indent=2) + '\n')


if __name__ == '__main__':
    main()
