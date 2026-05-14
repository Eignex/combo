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
statistics that drive each leaf's posterior update.

Using it requires three steps:

1. Declare the decision space — variables, sub-spaces, and constraints.
2. Pick a bandit algorithm and wire it to the compiled space.
3. Loop `choose()` → serve the configuration → `update()` with the observed reward.

## Decision space

The decision space describes the variables in the optimization problem in a tree
structure. Below is a simple top-list of media categories on a website where the
optimal configuration is learned over time from how each category performs:

```json
{
  "decisionSpace": {
    "name": "MediaPicker",

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
          "genre": ["Slasher", "Thriller", "MartialArts", "Crime", "Supernatural", "SuperHeroes", "Fantasy"]
        },
        "constraints": {
          "noSlasherForKids": "customerType == \"Child\" implies !movies.genre.contains(\"Slasher\")"
        }
      }
    },

    "constraints": {
      "betweenTwoAndFive": "|games| + |movies.genre| in [2, 5]"
    }
  }
}
```

Variable specs are tagged with `type` (klause / skema polymorphic JSON). Constraints
are shown as DSL strings — that parser is on the roadmap; on the wire today
constraints serialize as polymorphic AST trees.

A `multiple` is a set-valued variable: it expands at compile time to one boolean
indicator per label (`games.Shooter`, `games.Platform`, …). Inside `constraint { … }`
the matching Kotlin DSL reads `games.contains("Shooter")`, `games.containsAny(...)`,
`games.containsAll(...)`, `games.sizeLe(5)`, `games.sizeBetween(2, 5)`. Same
declaration surface as `nominal`, the only difference is "pick one" vs. "pick any
subset".

## Bandit

The flagship algorithm is **`RandomForestBandit`** — an online, constraint-aware
contextual bandit built on Hoeffding-bound decision trees (VFDT-style growth) that
each carry a *posterior distribution* at every leaf. At decision time the forest does
a guided descent against klause: each tree's split decisions are pinned as klause
assumptions, the local-search sampler proposes a feasible candidate consistent with
the pins, and if propagation reports the path infeasible the bandit backjumps to the
deepest conflicting decision rather than restarting. A complete `BacktrackSolver`
cascades behind the local search so a null sample means *definitive UNSAT*, not "the
sampler gave up under its budget" — the bandit's training data never carries
silently-biased exploration misses.

Exploration is per-leaf Thompson sampling on a Bayesian posterior over the leaf
reward, Oza–Russell online bagging diversifies the trees as updates stream in, and
Breiman-style mtry restricts each tree's split candidates to a random subspace. None
of this assumes a binary decision space: bools, integers, nominals, multi-selects, and
bucketed floats all carry their own typed split kinds, and constraints over them are
honored at every choose call by construction — there's no rejection-sample fallback
that quietly violates the model.

The bandit takes a compiled decision space plus a klause sampler that produces
feasible candidates:

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
val arm = bandit.choose() ?: return     // null only when no feasible sample exists
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

- **PGBM** — Probabilistic Gradient Boosting Machines
  ([Sprangers et al., 2021](https://arxiv.org/abs/2106.01682)): boosted trees with
  predictive distributions, slotting in next to the random forest as a stronger
  Thompson-sampling surrogate.
- **BoBandit** — Bayesian optimization with pluggable surrogates (linear, forest, GP)
  and acquisition functions (Thompson, UCB, EI), running over the same constrained
  decision space.
- **GP surrogate** — Gaussian-process surrogate for problems with a strong continuous
  component, starting with bucketed-float spaces and growing toward mixed-type kernels
  as use cases appear.
