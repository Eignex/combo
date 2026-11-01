<p align="center">
  <a href="https://eignex.com/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Eignex/.github/refs/heads/main/profile/banner-white.svg">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Eignex/.github/refs/heads/main/profile/banner.svg">
      <img alt="Eignex" src="https://raw.githubusercontent.com/Eignex/.github/refs/heads/main/profile/banner.svg" style="max-width: 100%; width: 22em;">
    </picture>
  </a>
</p>

# COMBO

A general-purpose toolkit for constrained decisions, statistics, and learning from
outcomes. Use Combo for resource allocation, service configuration, personalization,
or tuning expensive experiments.

- **Solve decisions:** find feasible configurations or optimize a known objective.
- **Track measurements:** maintain streaming statistics and use them in decisions.
- **Learn online:** adapt decisions with contextual bandits.
- **Optimize experiments:** use forest or Gaussian-process Bayesian optimization,
  with support for noisy outcomes and parallel trials.

Available as a Kotlin Multiplatform library and an HTTP API. The library targets
JVM, Linux x64, Linux ARM64, and macOS ARM64.

## Quick start

Start the HTTP server:

```sh
./gradlew :server:run
```

Create a model that minimizes worker count while meeting the current load:

```sh
curl -sS http://127.0.0.1:8080/v1/models \
  -H 'Content-Type: application/json' \
  -d '{
    "space": {
      "variables": {"workers": {"int": [1, 8]}},
      "context": {"load": {"int": [1, 8]}},
      "constraints": {
        "capacity": {"ge": [{"ref": "workers"}, {"ref": "load"}]}
      }
    },
    "objective": {"minimize": "workers"}
  }'
```

Use the returned `id` to request a decision:

```sh
curl -sS 'http://127.0.0.1:8080/v1/models/<id>/recommend' \
  -H 'Content-Type: application/json' \
  -d '{"given": {"load": 3}}'
```

The result is `workers: 3`. For measured objectives, propose configurations and
report their outcomes to learn better decisions over time.

## Documentation

- [HTTP API guide](server/README.md) — statistics, decisions, and observed outcomes.
- [Kotlin library guide](docs/guide.md) — decision spaces, bandits, and Bayesian optimization.
- [Benchmarks](combo-bench/README.md) — Kotlin workloads and eignex-lab integration.

Built on [Klause](https://github.com/Eignex/klause) for constraints and
[Kumulant](https://github.com/Eignex/kumulant) for streaming statistics and models.
