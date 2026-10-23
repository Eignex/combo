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

Combo optimizes typed Klause configurations under constraints. The library separates
algorithms by how they consume observations:

| Package | Data model | Algorithms |
| --- | --- | --- |
| `com.eignex.combo.bandit` | Streaming updates to model state | Online random forest, linear bandit, univariate bandits |
| `com.eignex.combo.bo` | Retained evaluation history and batch refitting | SMAC-style expected improvement, bootstrap forest Thompson sampling |
| `com.eignex.combo.shared` | Common learner interfaces, feature encoding, feedback, and feasible search | Used by both families |

All public packages use the `com.eignex.combo` prefix. Common learner contracts and
numeric feature encoders live in `com.eignex.combo.shared`.

Streaming learners keep sufficient statistics and incremental tree state. History-based
optimizers keep the evaluations themselves and can reconsider every observation when
rebuilding a model. This data-flow distinction applies whether an application targets
cumulative reward or the best final configuration.

Both families use the same decision-space schema and contextual variables.
[Klause](https://github.com/Eignex/klause) owns constraints and feasible witnesses;
[Kumulant](https://github.com/Eignex/kumulant) owns statistical models, posterior
sampling, batch CART forest fitting, and expected-improvement arithmetic.

Using it requires three steps:

1. Declare the decision space with its variables, sub-spaces, and constraints.
2. Pick a streaming learner or history-based optimizer and wire it to the compiled space.
3. Loop choose → serve the configuration → update with the observed reward.

## Decision space

The decision space describes what the optimizer can vary. In the example below the
bandit picks an LLM agent setup (model, temperature, prompt style, and tools) per
request, conditioned on the task type and user tier, and learns from whether the run
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

Every choose call returns a feasible configuration. Variables are typed: bool, int,
float, nominal (pick one), or multiple (pick any subset).

## Bandit

The flagship is the random forest bandit: an online, constraint-aware contextual
bandit built on Hoeffding-bound decision trees, each carrying a Bayesian posterior at
every leaf. At decision time it combines candidate sampling with the original
cross-tree decision algorithm, reusing feasible witnesses and asking Klause to repair
or backtrack only when necessary. Context values stay fixed throughout the search.

Exploration is per-leaf Thompson sampling on a Bayesian posterior over the leaf
reward, with online bagging diversifying the trees as updates stream in and mtry
restricting each tree's split candidates to a random subspace. Bools, integers,
nominals, multi-selects, and floats all carry their own typed split kinds,
and every constraint is honored at every choose call by construction, with no
rejection-sample fallback that might quietly violate the model.

The default builder wires the forest to a hybrid Klause search:

```kotlin
val space = MyDecisionSpace().compileSpace()
val bandit = RandomForestBandit.build(
    space = space,
    nbrTrees = 25,
    candidateBatchSize = 8,
    poolCapacity = 64,
    randomSeed = 42,
)
```

Kumulant supplies the growing trees and Normal-Gamma Thompson policy. Each round
samples a batch of feasible candidates and scores their merged leaf posteriors.
Candidates reaching the same leaf tuple share one posterior draw for that round;
adding correlated trees does not multiply the effective observation count.

The original cross-tree descent then pins Boolean and nominal decisions. Compatible
cached witnesses answer those checks without a solver call. A missing witness triggers
a local-search repair seeded from a nearby solution, followed by bounded backtracking
when repair is inconclusive. The best feasible candidate is retained even if a
preferred branch exhausts its search budget. Integer and float splits also affect
candidate scoring; continuous candidate generation uses solver-produced feasible
vertices in random linear directions, preserving Klause's values without a grid.

Defaults are 2,000 local-search flips and 10,000 backtracking decisions per request;
pass `localSearch = LocalSearchParams(...)` and `backtrack = BacktrackParams(...)`
to tune them. A search budget miss is `Unknown`, not an infeasibility proof.
`HybridDescentSession.lastResult`, `poolHits`, `proposalCalls`, and `backtrackCalls`
are available through `bandit.descentSession` for inspecting the search. The existing
callback-based builder remains available for other Klause engines and strategies.

The forest's `regression` is Kumulant's `RandomForestRegressionStat`: it owns the
trees, seed streams, online bagging, and immutable statistical snapshots. Combo's
`TreeVectorLayout` supplies a read-only numeric view and maps learned predicates
back to Klause expressions. Numeric coordinates preserve declared units; expression
predicates use Boolean feature slots. Both linear and forest bandits use
`com.eignex.combo.shared` for training, prediction, reward/error tracking, and model
snapshot handling, witness caching, and Klause session adapters. The forest uses
the shared hybrid session; the linear bandit uses its supplied objective optimizer.
Tree and forest `exportData`/`importData` transfer actual Kumulant snapshots;
import merges evidence and requires the same ordered predicate layout.

Then it's a choose → serve → update loop:

```kotlin
val arm = bandit.choose() ?: return     // no feasible witness found within the budgets
serve(arm)                              // hand the configuration to the user
val reward = observe()                  // measure clicks, sales, latency, …
bandit.update(arm, reward)
```

Combo extends Klause's variables with context metadata. Context variables use the
same types and constraints as decision variables; the caller supplies their values,
and each choose call pins them through Klause assumptions. Floats retain Klause's
semantics and decoded values. Combo adds no grid or post-solve perturbation.

Use `bandit.choose(context { ... })`, `bandit.update(arm, context, reward)`, and
`bandit.optimal(context)` for contextual optimization. The same context remains a
hard constraint during candidate generation, branch repair, and solver fallback.

End-to-end examples covering cascade-with-backtrack, Thompson sampling on linear
models, and context-conditioned arms live under src/jvmTest/kotlin/com/eignex/combo.

## History-based optimization

```kotlin
import com.eignex.combo.bo.InMemoryObservationStore
import com.eignex.combo.bo.RandomForestOptimizer

val history = InMemoryObservationStore()
val optimizer = RandomForestOptimizer.smac(space, store = history)
// Alternative: RandomForestOptimizer.bootstrapThompson(space, store = history)

val configuration = optimizer.chooseOrThrow()
optimizer.update(configuration, evaluate(configuration))
val recommendation = optimizer.optimalOrThrow()
```

Both algorithms use Kumulant's batch CART forest fitter, which finds numeric split
thresholds from the observed data. Continuous values retain their declared units.

- **SMAC-style expected improvement** refits a forest when the store revision changes.
  It models the distribution of tree predictions as Gaussian and maximizes expected
  improvement. This implements the forest/EI pattern described in
  [SMAC](https://www.jmlr.org/papers/v23/21-0888.html), using Klause's feasible candidate
  search. It does not implement SMAC's racing, censored trials, or multi-fidelity scheduler.
- **Bootstrap Thompson sampling** resamples the full evaluation history with
  replacement and fits a fresh random forest for each exploratory choice. The sampled
  model stays fixed throughout that choice. This follows the bootstrap-model sampling
  approach in [TreeBootstrap](https://www.auai.org/uai2017/proceedings/papers/171.pdf).

`ForestOptimizerConfig` controls the batch forest, initial random observations,
random exploration probability, and EI margin. `optimal` always uses the mean of a
forest fitted to the full history. Contextual choose/update overloads match those of
the streaming forest; observations across contexts train a shared response model.
Historical configurations also enter candidate ranking after Klause checks their
feasibility, preserving observed interior points. Fresh continuous candidates use
the hybrid session's random linear directions, so global acquisition maximization
and general interior exploration require a richer custom candidate generator.

`ObservationStore` is the storage boundary. Its append operation advances a revision,
and `snapshot()` returns a stable, ordered history. The default in-memory store retains
all observations without eviction. A persistent implementation can use the same
interface. Stores belong to one decision-space schema. `train` records off-policy
evaluations without reward/error bookkeeping; zero weight is a no-op and negative
weights are rejected. Import/export transfers schema-bound history snapshots.
Configuring pre/post-training error sinks can trigger additional fits during updates.

## Building

The build uses Kbuild 1.3.3, Kotlin 2.4.10, Gradle 9.7.0, and a JDK 25 toolchain
that Gradle provisions automatically. Klause and Kumulant use their latest published
snapshots from October 23, 2026. Exact platform artifact versions are pinned in
`build.gradle.kts`, alongside a shared Koblas version that resolves their conflicting
backend pins. The configured targets are JVM and Linux
x64, matching Klause's available snapshots; Combo's implementation is currently JVM.

Build and validate with the published dependencies:

```sh
./gradlew jvmTest
./gradlew check lintDocs
```

For sibling development, `-Pkumulant.local` substitutes `../kumulant`.
Use `-Pdependencies.local` to develop against both `../klause` and `../kumulant`.
The Detekt baseline records existing documentation and style debt surfaced by the
new Kbuild checks; new findings remain checked. ABI references live under `api/`.

JVM applications using the vector backend should enable
`--add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED`; the test
tasks already supply these flags.

## Roadmap

- **PGBM**: Probabilistic Gradient Boosting Machines
  ([Sprangers et al., 2021](https://arxiv.org/abs/2106.01682)): boosted trees with
  predictive distributions, slotting in next to the random forest as a stronger
  Thompson-sampling surrogate.
- **Additional history-based surrogates**: extend the shared store/refit lifecycle
  to batch linear and Gaussian-process models, with more acquisition functions.
- **GP surrogate**: Gaussian-process surrogate for problems with a strong continuous
  component, using Klause's continuous variables and growing toward mixed-type kernels
  as use cases appear.
