#!/usr/bin/env python3
"""Evaluate Combo's captured posteriors with BoTorch, holding model and base normals fixed."""
import argparse
import csv
import json
import gzip
import warnings
from importlib.metadata import version
from pathlib import Path
import torch
import gpytorch
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('benchmarks/gp-refinement/frozen.json.gz'))
    parser.add_argument('--output', type=Path, default=Path('build/reports/gp-refinement/reference.csv'))
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    warnings.filterwarnings('ignore', category=Warning, module='botorch')
    rows = []
    payload = gzip.decompress(args.input.read_bytes()).decode() if args.input.suffix == '.gz' else args.input.read_text()
    for fixture in json.loads(payload):
        model = FrozenPosterior(fixture)
        fixed = CapturedSampler(fixture['normals'])
        high = SobolQMCNormalSampler(sample_shape=torch.Size([16384]), seed=8123)
        baseline = torch.arange(fixture['baseline']).double().unsqueeze(-1)
        noisy = fixture['id'].endswith('qnei')
        def acquisition(sampler, log=False):
            if noisy:
                cls = qLogNoisyExpectedImprovement if log else qNoisyExpectedImprovement
                return cls(model, baseline, sampler=sampler, prune_baseline=False, cache_root=False)
            cls = qLogExpectedImprovement if log else qExpectedImprovement
            return cls(model, best_f=fixture['incumbent'], sampler=sampler)
        direct, precise, logarithmic = acquisition(fixed), acquisition(high), acquisition(fixed, log=True)
        for case in fixture['sets']:
            x = torch.tensor(case['indices'], dtype=torch.float64).reshape(1, -1, 1)
            with torch.no_grad(), gpytorch.settings.fast_computations(covar_root_decomposition=False):
                actual, reference, log_value = direct(x).item(), precise(x).item(), logarithmic(x).item()
            assert abs(actual - case['legacy']) < 1e-9, (fixture['id'], case['indices'], actual, case['legacy'])
            rows.append(dict(fixture=fixture['id'], q=len(case['indices']), indices=';'.join(map(str, case['indices'])),
                             legacy=case['legacy'], botorch_fixed=actual, botorch_high=reference, botorch_log=log_value))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix('.versions.json').write_text(json.dumps({
        name: version(name) for name in ('torch', 'botorch', 'gpytorch', 'numpy', 'scipy')}, indent=2) + '\n')
    print(f'Compared {len(rows)} values on fixed GP posteriors; captured-normal agreement < 1e-9.')


class FrozenPosterior(Model):
    def __init__(self, fixture):
        super().__init__()
        self.register_buffer('means', torch.tensor(fixture['means']))
        self.register_buffer('covariance', torch.tensor(fixture['covariance']))

    @property
    def num_outputs(self):
        return 1

    @property
    def batch_shape(self):
        return torch.Size([])

    def posterior(self, X, output_indices=None, observation_noise=False, posterior_transform=None):
        indices = X[..., 0].long()
        means = self.means[indices]
        cov = self.covariance[indices.unsqueeze(-1), indices.unsqueeze(-2)]
        result = GPyTorchPosterior(gpytorch.distributions.MultivariateNormal(means, cov))
        return posterior_transform(result) if posterior_transform else result


class CapturedSampler(MCSampler):
    def __init__(self, normals):
        super().__init__(sample_shape=torch.Size([256]))
        self.register_buffer('captured', torch.tensor(normals).T)

    def forward(self, posterior):
        shape = posterior.base_sample_shape
        n = shape[-1]
        base = self.captured[:, :n].reshape(256, *((1,) * (len(shape) - 1)), n)
        return posterior.rsample_from_base_samples(self.sample_shape, base.expand(self.sample_shape + shape))


if __name__ == '__main__':
    main()
