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

The decision space describes the variables the optimizer can choose between. Below is
an LLM-agent configurator: at every request the bandit decides which model, sampling
temperature, prompt strategy, and tool set to dispatch — conditioned on what kind of
task the user submitted and which tier they're on — and learns from whether the run
succeeded.

```json
{
  "decisionSpace": {
    "name": "AgentPolicy",

    "context": {
      "taskType": { "type": "nominal", "labels": ["coding", "writing", "research", "casual"] },
      "userTier": { "type": "nominal", "labels": ["free", "pro", "enterprise"] }
    },

    "variables": {
      "model":       { "type": "nominal", "labels": ["haiku", "sonnet", "opus"] },
      "temperature": { "type": "float",   "min": 0.0, "max": 1.0 },
      "maxTokens":   { "type": "int",     "min": 256, "max": 8192 },
      "promptStyle": { "type": "nominal", "labels": ["terse", "detailed", "chainOfThought"] },
      "tools": {
        "type": "multiple",
        "labels": ["web_search", "code_exec", "file_read", "bash"]
      }
    },

    "constraints": {
      "freeTierCantUseOpus":  "userTier == \"free\" implies model != \"opus\"",
      "codeExecNeedsBash":    "tools.contains(\"code_exec\") implies tools.contains(\"bash\")",
      "codingTasksNeedTools": "taskType == \"coding\" implies (tools.contains(\"code_exec\") or tools.contains(\"file_read\"))",
      "casualKeepsItCheap":   "taskType == \"casual\" implies (maxTokens <= 1024 and model != \"opus\")"
    }
  }
}
```

Every choose call returns a feasible configuration honoring all constraints — there
is no rejection-sample fallback that might quietly serve an `opus` model to a free-tier
user. Each variable carries its own `type` tag (`bool`, `int`, `nominal`, `float`,
`multiple`).

A `multiple` is a set-valued variable — `tools` here picks any subset of the four
labels. In the Kotlin DSL it reads `tools.contains("web_search")`,
`tools.containsAll("code_exec", "bash")`, `tools.sizeLe(2)`, `tools.sizeBetween(1, 3)`
— same declaration surface as `nominal`, the difference is "pick one" vs. "pick any
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
