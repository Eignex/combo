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

Combo is a library for **Constraint-Oriented Multi-variate Bandit Optimization** of
software parameters: each user gets their own configuration drawn from a constrained
decision space, and the reward signal (clicks, sales, latency, …) shapes future
configurations in real time. Supported algorithms are random forest (Thompson sampling
on a per-leaf posterior), GLM (linear Thompson sampling with a Bayesian linear model),
and a univariate multi-armed bandit for the no-context case.

Combo sits on top of two sibling libraries:
[klause](https://github.com/Eignex/klause) handles the constraint solving and
sampling, and [kumulant](https://github.com/Eignex/kumulant) provides the streaming
statistics and posterior distributions.

Using it requires three steps:

1. Declare the decision space — variables, sub-spaces, and constraints.
2. Pick a bandit algorithm and wire it to the compiled space.
3. Loop `chooseOrThrow()` → serve the configuration → `update()` with the observed reward.

## Decision space

The decision space describes the variables in the optimization problem in a tree
structure. Below is a simple top-list of media categories on a website where the
optimal configuration is learned over time from how each category performs:

```json
{
  "decisionSpace": {
    "context": {
      "displayWidth": { "type": "int", "min": 640, "max": 1920 },
      "customerType": { "type": "nominal", "labels": ["Child", "Company", "Person"] }
    },

    "multiples": {
      "games": ["Shooter", "Platform", "Sports", "Action", "Adventure", "Strategy"]
    },

    "spaces": {
      "movies": {
        "multiples": {
          "horror": ["Slasher", "Splatter", "Zombie"],
          "action": ["Thriller", "MartialArts", "Crime"],
          "sciFi":  ["Supernatural", "SuperHeroes", "Fantasy"]
        },
        "constraints": [
          "customerType == \"Child\" implies |horror| == 0"
        ]
      }
    },

    "constraints": [
      "|games| + |movies.horror| + |movies.action| + |movies.sciFi| in [2, 5]"
    ]
  }
}
```

A `multiple` is a set-valued variable: it expands at compile time to one boolean
indicator per label (`games.Shooter`, `games.Platform`, …). Inside `constraint { … }`
the matching Kotlin DSL reads `games.contains("Shooter")`, `games.containsAny(...)`,
`games.containsAll(...)`, `games.sizeLe(5)`, `games.sizeBetween(2, 5)`. Same
declaration surface as `nominal`, the only difference is "pick one" vs. "pick any
subset".

## Bandit

Random forest is a good default — robust to bad tuning, copes with both discrete and
mixed search spaces. The bandit takes a compiled decision space plus a klause sampler
that produces feasible candidates:

```kotlin
val space = MyDecisionSpace().compileSpace()
val solver = LocalSearchSolver(space.compiled.problem)

val bandit = RandomForestBandit.build(
    space = space,
    policy = ThompsonSampling(),
    proposeSample = { rng, assumptions ->
        solver.sample(LocalSearchParams(randomSeed = rng.nextLong(), assumptions = assumptions))
    },
    nbrTrees = 25,
)
```

Then it's a `choose` → serve → `update` loop:

```kotlin
val arm = bandit.chooseOrThrow()        // BanditSample drawn from the posterior
serve(arm)                              // hand the configuration to the user
val reward = observe()                  // measure clicks, sales, latency, …
bandit.update(arm, reward)
```

Context (per-call features supplied by the caller — `customerType`, `displayWidth` in
the example above) is passed through `Context { set(handle, value) }` and steers the
bandit's policy without entering the decision space itself.

See `src/jvmTest/kotlin/combo/bandit` for end-to-end examples — `RandomForestBanditTest`
and `LinearBanditTest` show the cascade-with-backtrack pattern, Thompson sampling on
linear models, and context-conditioned arms.

## Roadmap

Three algorithm slices are planned next, each reusing the existing decision-space /
sampler / projection infrastructure rather than introducing a parallel stack.

### PGBM — Probabilistic Gradient Boosting Machines

[PGBM (Sprangers et al., 2021)](https://arxiv.org/abs/2106.01682) layers boosted trees
that emit a *predictive distribution* per leaf (mean and variance, propagated through
the boosting sum) instead of point estimates. That gives Thompson sampling on the
output natively, so the surrounding bandit machinery is identical to
`RandomForestBandit`: same `Tree<R>` structure, same `defaultSplitCandidates`, same
`LeafMaterialization` pinning. The only new piece is the boosting update rule —
gradient/Hessian per leaf, additive contributions across trees — and a distribution-
sample policy in place of bagged-tree averaging. Lands as
`combo.bandit.pgbm.PgbmBandit` next to the random forest, sharing the `combo.bandit.dt`
tree primitives.

### BoBandit — Bayesian optimization with pluggable surrogates

A constrained-BO loop where klause provides the *feasibility* and a swappable
**surrogate** provides the *posterior*. Sketch:

```kotlin
BoBandit<S : Surrogate>(
    space: CompiledDecisionSpace,
    surrogate: S,                       // LinearSurrogate | ForestSurrogate | GpSurrogate
    acquisition: Acquisition,           // Thompson | UCB(β) | ExpectedImprovement
    inner: InnerSolver,                 // LinearMinimize(z3) | SampleAndScore(localSearch, k)
    rng: Random,
)
```

Two inner-optimizer modes, chosen by acquisition shape:

1. **Linear closed form** — Thompson on a linear surrogate. Draw a weight vector from
   the posterior, build a klause `LinearObjective`, hand to `BacktrackSolver.minimize`
   (or `Z3Sampler.minimize` for hard combinatorial cases). One exact call per round.
2. **Sample-and-score** — UCB / EI / nonlinear surrogate. Draw `k` feasible candidates
   via `Sampler.samples(...)` with `minHammingDistance` for diversity, score each
   through the surrogate's posterior, pick argmax. SMAC-style when paired with a forest
   surrogate; matches the textbook BO recipe when paired with a linear or GP one.

`LinearSurrogate` falls out of the existing `CovarianceLinearModel`; `ForestSurrogate`
wraps the random forest's per-leaf posterior. Both should land before the GP work.

### GP surrogate over mixed spaces

A `GpSurrogate` on top of `BoBandit` for problems with a strong continuous component.
The hard part is the kernel: combo's variables are mixed bool/int/nominal/bucketed-float,
and general-purpose GP kernels for that combination are research-grade (categorical
kernels, ARD over heterogeneous types, marginal-likelihood tuning under constraints).
The plan is to start with a bucketed-float-only GP — Cholesky already lives in
`combo.bandit.util` for the linear-Bayes posterior, so the lift is just kernel +
hyperparameter optimization — and add categorical / hierarchical kernels as concrete
use cases appear.
