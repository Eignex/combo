# Eignex Server

An API-only Ktor application for statistics, constrained decisions, and learning
from outcomes. Everything lives under `/v1/models`. Clients describe their model
and use its capabilities; they do not select a library or an engine.

The `:server` application and `:combo` library are separate Gradle modules in the
same repository. Start the server with `./gradlew :server:run`.
The default address is `127.0.0.1:8080`; configure it with `EIGNEX_SERVER_HOST` and
`EIGNEX_SERVER_PORT`. Gradle provisions JDK 25 for development.
`./gradlew :server:installDist` creates `server/build/install/eignex-server`;
its launcher also needs JDK 25.

## Shared lifecycle

`POST /v1/models` creates an immutable definition with retained, mutable state.
It returns `201`, `{"id":"..."}`, and a `Location` header.

| Request | Purpose |
| --- | --- |
| `GET /v1/models` | List model identifiers |
| `GET /v1/models/{id}` | Read `definition`, `capabilities`, and `state` |
| `POST /v1/models/{id}/evaluate` | Check a supplied decision and evaluate its objective |
| `POST /v1/models/{id}/recommend` | Return the best known decision |
| `POST /v1/models/{id}/propose` | Select decisions to gather evidence |
| `POST /v1/models/{id}/observe` | Add an observation |
| `DELETE /v1/models/{id}/proposals/{proposal}` | Cancel a pending proposal |
| `POST /v1/models/{id}/reset` | Reset a standalone statistic or bandit |
| `DELETE /v1/models/{id}` | Release the model |

Each model advertises its supported operations in `capabilities`. A statistic can
observe; a bandit can also propose; a decision model can evaluate and recommend.
An observed objective adds proposals and feedback to that same decision model.
`GET /health` reports process health.

## Native problems

`problem` embeds Klause's canonical `ProblemJson` document directly. It retains every
registered native factor, sparse and open domains, 64-bit declarations, exact factor
constants, implied-factor flags, symmetry declarations, resource settings, and source
semantics. Combo delegates decoding and encoding to Klause. `space` remains the named
schema convenience; a definition selects either `problem` or `space`.

For example, constrain a sparse integer column using a native table:

```json
{
  "problem": {
    "ints": [{"lower": 0, "upper": 10, "values": [[0, 2], [9, 10]]}],
    "factors": [{"table": {"vars": [0], "tuples": [[1], [9]]}}]
  },
  "objective": {"minimize": "ints.0"}
}
```

Native columns are addressed as `bools.0`, `ints.0`, and `reals.0`, in their separate
native namespaces. `/recommend` returns `ints.0 = 1` here. `/evaluate` accepts
`{"values":{"ints.0":9}}`. Any subset of native columns can be held fixed using
`given`; `values` supplies the remaining columns for evaluation. The server returns
the supplied columns under `given` and the others under `values`.

To learn a measured objective over exactly the same problem, replace `objective`
with `{"maximize":"reward"}` and add `"outcome":"reward"`. `/propose` and
`/observe` then use the existing forest learner and retained proposal lifecycle.
The existing `learning` configuration also supports the native linear learner.
Native constraints always remain attached during proposal search.

A named statistic can supply an existing numeric column directly:

```json
{
  "problem": {"ints":[{"lower":0,"upper":8},{"lower":0,"upper":8}],
    "factors":[{"linear":{"vars":[0,1],"coefficients":[1,-1],"op":"GE","bound":0}}]},
  "objective": {"minimize":"ints.0"},
  "statistics": {"demand":"mean"},
  "inputs": {"ints.1":{"statistic":"demand","value":"mean"}}
}
```

After observing demand, each decision freezes its mean and pins `ints.1` with a
native equality. This works for known objectives and learned outcomes. Native
bindings select an explicit scalar projection; interval interpretation stays in
the named schema DSL. Callers cannot override bound columns.

Imported problems use Klause's source strategy. Open domains do not acquire finite
boxes, and native `unbounded`, `unattained_limit`, and `unknown` verdicts remain
distinct. Support for a mathematical fragment depends on Klause's selected native
route; accepting its document does not promise a proof for every factor combination.
Known-objective solving retains native exact assignments. Integer responses beyond
`Long` use decimal strings; rational reals that are not exactly binary64 use fraction
strings such as `"1/3"`. These values can be submitted again for evaluation.
Irrational or symbolic real responses expose a tagged `{"approximation":...}`;
that tag is informational, not an exact input value. Learned numerical features use
Double approximations. Native candidate generation preserves exact solver witnesses
rather than reconstructing them through those features.

## A moving average

Create a model:

```http
POST /v1/models
Content-Type: application/json

{"stat":{"ewma":{"alpha":0.1}}}
```

Feed the returned identifier an observation:

```http
POST /v1/models/{id}/observe
Content-Type: application/json

{"value":42.0}
```

The response is the current snapshot: `{"totalWeights":1.0,"mean":42.0}`.
Reading the model returns that snapshot under `state`.

To add one automatically to every observation, compose the statistic:

```json
{
  "stat": {
    "transform": {
      "expr": {"add":[{"input":"x"},1]},
      "inner": {"ewma":{"alpha":0.1}}
    }
  }
}
```

The same observation then produces a mean of `43.0`. Here `{"input":"x"}` names the incoming value in the canonical expression format;
expressions are structured JSON.
The server passes `stat` directly to the native serializable specification.
Filters, weighting, groups, and other compositions use that same format.

## Decisions with a known objective

Declare decision variables and context supplied at request time. An objective
names the variable to minimize or maximize:

```http
POST /v1/models
Content-Type: application/json

{
  "space": {
    "variables": {"workers":{"int":[1,8]}},
    "context": {"load":{"int":[1,8]}},
    "constraints": {
      "capacity": {"ge":[{"ref":"workers"},{"ref":"load"}]}
    }
  },
  "objective": {"minimize":"workers"}
}
```

This requires `workers >= load`. Recommend with the current load:

```http
POST /v1/models/{id}/recommend
Content-Type: application/json

{"given":{"load":3}}
```

```json
{
  "status":"optimal",
  "values":{"workers":3},
  "given":{"load":3},
  "objective":{"name":"workers","value":3}
}
```

To evaluate a supplied configuration, send
`{"given":{"load":3},"values":{"workers":4}}` to `/evaluate`.
It returns `status: "feasible"` and the objective value; `workers: 2` returns
`status: "infeasible"`. Every request must supply every context variable and,
when evaluating, every decision variable. Context is fixed for that request.

Linear relationships can define objective variables too. For example, declaring
`"cost":{"int":[0,20]}` and adding the named constraint
`{"eq":[{"ref":"cost"},{"mul":[2,{"ref":"workers"}]}]}`
allows `{"minimize":"cost"}`. There is no expression string language.

Without an objective, `/recommend` returns any feasible assignment. Search results
distinguish `feasible`, `optimal`, `best_found`, `infeasible`, `unbounded`, and
`unknown`. Real-valued optimization conservatively reports `best_found` rather
than claiming a proof of continuous optimality.

## Decisions with an observed objective

When an objective must be measured, name it as an `outcome` outside the search
domain:

```http
POST /v1/models
Content-Type: application/json

{
  "space": {"variables":{"workers":{"int":[1,8]}}},
  "outcome": "latency",
  "objective": {"minimize":"latency"},
  "stat": {"ewma":{"alpha":0.1}},
  "seed": 42
}
```

The optional `stat` tracks observed objective values. Request experiments:

```http
POST /v1/models/{id}/propose
Content-Type: application/json

{"count":2}
```

```json
{
  "status":"ready",
  "proposals":[
    {"id":0,"status":"pending","values":{"workers":3},"given":{}},
    {"id":1,"status":"pending","values":{"workers":6},"given":{}}
  ]
}
```

The values above are illustrative. Pending proposals reserve their configuration.
Execute them outside the server and report the measured outcome by proposal ID:

```http
POST /v1/models/{id}/observe
Content-Type: application/json

{"proposal":0,"values":{"latency":82.4}}
```

Feedback may arrive out of order. The server retains the original decision and
context, records feedback once, and updates the optional statistic. Model state
contains `observations`, `proposals`, and the optional `stat` snapshot.
Observation requests can also supply positive finite `weight` and nonnegative
finite `noiseVariance`.

Use `/recommend` with the same `given` shape as a known objective. Its result has
`status: "estimated"` and `objective: {"name":"latency","estimate":...}`.
`/evaluate` similarly predicts the objective for supplied feasible decisions.
These operations require at least one observation; otherwise they return `409`
with `insufficient_evidence`. Estimates do not claim proven optimality.

Cancel a proposal to release its reservation without adding evidence. Closed
proposals reject further feedback with `409`. Batch proposals return `ready`,
`partial`, or `unavailable`, with up to 64 proposals per request.

## Statistics and bandits

The `stat` field also accepts discrete, paired, vector, and regression statistics,
as well as ordinary and contextual bandits. Send one observation to `/observe`:

| Model input | Observation |
| --- | --- |
| Series | `{"value":42.0}` |
| Discrete | `{"value":42}` |
| Paired | `{"x":1.0,"y":2.0}` |
| Vector | `{"x":[1.0,2.0]}` |
| Regression | `{"x":[1.0,2.0],"y":3.0}` |
| Ordinary bandit | `{"arm":0,"reward":0.8}` |
| Contextual bandit | `{"arm":0,"x":[1.0,2.0],"reward":0.8}` |

These observations accept finite `weight`, defaulting to `1.0`; zero and negative
weights follow the statistic's semantics. Stat observations also accept
`timestampNanos`; otherwise the server uses its monotonic clock. Reads can specify
`GET /v1/models/{id}?timestampNanos=...` in that same clock domain.
Undefined or non-finite statistical outputs are JSON `null`.

For example, create a two-arm bandit:

```json
{"stat":{"multiArmed":{"nbrArms":2,"policy":{"greedy":{}}}},"seed":42}
```

`POST /v1/models/{id}/propose` with `{}` returns
`{"status":"ready","proposals":[{"values":{"arm":0}}]}`.
Report its reward through `/observe`. Arm selection does not reserve a
configuration, so these proposals have no identifier.

A contextual example:

```json
{
  "stat": {
    "regressionContextual": {
      "nbrArms":2,
      "regression":{"bayesianRegression":{"featureSize":2}},
      "posterior":{"linUcb":{}},
      "exploration":1.0
    }
  }
}
```

Propose with `{"x":[1.0,2.0]}` and observe with that same context, the chosen arm,
and the reward. Bandit state includes an `arms` array; pooled regressors also
include `global`. Tracked bandits and grouped statistics retain their native
snapshot structure. `/reset` restores a standalone stat or bandit to its baseline.

## Statistical inputs in decision constraints

A model can own named native statistics and bind their outputs into Klause
constraints. Input names are references, not additional decision variables:

```json
{
  "space": {
    "variables": {"capacity":{"int":[0,200]}},
    "constraints": {
      "enough": {"ge":[{"ref":"capacity"},{"ref":"latency"}]}
    }
  },
  "objective": {"minimize":"capacity"},
  "statistics": {"timings":{"band":{"inner":"variance","k":1}}},
  "inputs": {"latency":{"statistic":"timings"}}
}
```

Observe the statistic using its existing native input fields:

```json
{"statistic":"timings","observation":{"value":80}}
```

After observations of 80 and 120, this native band has lower/upper outputs of
80 and 120. A recommendation chooses capacity 120, because `capacity >= latency`
must hold throughout the interval. The response includes the selected projections
under `inputs`; only capacity appears under decision `values`. Each solve reads
one snapshot per named statistic, then creates fixed native declarations. The
solver cannot choose a convenient statistical value, and callers cannot override
inputs through `given` or `values`.

Bare interval references support comparisons and linear integer arithmetic with
known integer coefficients, including negative coefficients and cancellation.
Equality must hold at both endpoints; inequality must exclude the entire interval.
Conjunctions and implication with a decision-only condition are supported.
Uncertain disjunctions, compound negations, and nonlinear arithmetic require
explicit projections. For example:

```json
{"le":[{"mul":[{"ref":"latency.upper"},{"ref":"workers"}]},1000]}
```

A projection selects its same-named native numeric output. Optional aliases use
structured paths: `"projections":{"mean":["center"]}` exposes a band's center as
`latency.mean`. `path` selects a nested result object, so multiple inputs can view
one shared joint statistic; its paired/vector observation is still updated once.
All views use the same snapshot. Correlations are not used to tighten the interval
bounds in this version.

For a fixed scalar input, select its output with `value`:

```json
{
  "statistics":{"latency":{"ewma":{"alpha":0.1}}},
  "inputs":{"latency":{"statistic":"latency","value":"mean"}}
}
```

The snippet above supplies fields to a model with a decision space. Explicit
projections can also be named objectives, such as `"minimize":"latency.mean"`.
Missing, nonfinite, or inverted interval outputs are errors. Integer expressions
require exactly representable 32-bit integer projections; values are never rounded.
Fractional projections work with native `floatLinear` constraints and the real DSL.

The Kotlin handles `StatisticalInt("latency")` and `StatisticalReal("latency")`
implement Klause's existing term interfaces. Their `upper`, `lower`, `mean`,
`median`, and `projection(name)` members produce ordinary Klause references.
They introduce no serialized expression nodes. For example, using Klause's
existing operators:

```kotlin
val latency = StatisticalReal("latency")
val constraint = latency lt 100.0
```

The binding must be declared in `ModelDefinition.inputs`. Statistical inputs
support feasibility, known objectives, and learned objectives as described below.
Candidate-dependent statistical predictions and sampling statistical input
distributions are not implemented. Native bands retain Kumulant's own meaning; the
integration does not interpret a scale multiplier as a calibrated confidence level.
Before solving, each referenced statistic needs a positive-weight observation
(`409 insufficient_evidence` otherwise).

## Learning with statistical context

A learned objective can use statistics as context. Declare the context's native
domain and bind a scalar projection to it:

```json
{
  "space": {
    "variables": {"workers":{"int":[1,8]}},
    "context": {"load":{"int":[1,8]}},
    "constraints": {
      "capacity":{"ge":[{"ref":"workers"},{"ref":"load"}]}
    },
    "interactions": [{
      "name":"workers_load",
      "lhs":{"kind":"IntContext","path":["context","load"]},
      "rhs":{"kind":"IntDecision","path":["workers"]}
    }]
  },
  "statistics": {"demand":"mean"},
  "inputs": {"load":{"statistic":"demand","value":"mean"}},
  "outcome":"reward",
  "objective":{"maximize":"reward"},
  "learning": {
    "regression":{"bayesianRegression":{"featureSize":3}},
    "posterior":"multivariateGaussian"
  }
}
```

Here the features are workers, load, and their interaction. The regression and
posterior use Kumulant's existing specifications and canonical serialization.
The regression contains its prior; `posterior` independently specifies how to draw
coefficients. This initial linear option supports Bayesian regression with the
multivariate Gaussian sampler. The declared feature count must match the native
projection, including contexts and interactions. `exploration` defaults to `1.0`.
This trial adapter currently requires linear numeric scales.
Omitting `learning` retains the default history-based optimizer.

Feed demand with `{"statistic":"demand","observation":{"value":4}}`, then
call `/propose` with `{}`. The statistic supplies load; callers must omit bound
contexts from `given`. Other contexts still need to be supplied. The selected
value appears as `"inputs":{"load.mean":4}` in the result.

Each request reads one snapshot per statistic, and every proposal in its batch
uses the same resolved context. Linear proposals draw a coefficient vector once
per proposal and optimize under that draw. Outstanding reservations are excluded;
when exclusions are present, selection ranks a bounded pool of feasible candidates.
Recommendations optimize the posterior mean. Both use the existing native linear
feature projection, including context interactions.

Feedback trains on the proposal's retained context, even if statistics changed
before the reward arrived. Reading or updating a statistic does not reset the
learner or add a reward observation. Statistical input values are never sampled by
the coefficient policy. They remain pinned in constraints as well as predictions.

Learned bindings require a scalar `value` inside a declared numeric context
domain. Integer contexts require exact integers. Native float declarations accept
fractional values: omit `buckets` for continuous real semantics, or specify an
grid request. Klause chooses the effective representation for each component.
Values outside the domain or off an effective finite grid return `400` without
issuing a proposal; retained real coordinates accept fractional values unchanged. Linear feedback accepts weights
but rejects per-observation `noiseVariance`, which the native linear update API
does not support. Closed proposals continue to reject repeated feedback.

## Initial scope and operation limits

The public contract is `com.eignex.combo.model.ModelDefinition` in `:combo`.
The server consumes that contract directly. Its `space` is the existing
`DecisionSpaceDef`: native Klause `VarSpec` declarations and `BoolExpr` constraints,
including structured arithmetic, logical expressions, global constraints, and
nested spaces. The server has no separate variable or constraint DTO. Statistics
use Kumulant's canonical JSON, including native expression bindings.

The current HTTP runtime supports scalar decisions: `"bool"`, `{"int":[min,max]}`,
`{"float":{"min":0.0,"max":1.0}}`, and `{"nominal":["a","b"]}`.
Nested assignments use qualified names, such as `child.workers`. Context supports
bounded integers, Booleans, and native floats declared at the root. Every context must be supplied;
optional context is not yet supported. Real values are pinned with native equality rows before solving or validating
an assignment, so constraints are checked against the requested real values.
Real-valued context interactions use the same native feature projection as
integer context interactions. The Kotlin DSL exposes `realVar` / `contextReal`
for continuous declarations and `floatVar` / `contextFloat` with optional grids;
both retain Klause's canonical float JSON. An observed objective is one finite real outcome;
constraints cannot depend on it. Attached outcome statistics must accept scalar
observations. These runtime limits do not introduce another serialization format.

Statistical bindings are owned by Combo and lower into native Klause declarations.
Klause and Kumulant are consumed unchanged; the server defines no binding DTO or
expression vocabulary.

Malformed input and unsupported operations return `400`; unknown models and
proposals return `404`. Saturation returns `503` with `Retry-After`.
Errors have the shape `{code, message}`. Infeasibility is a normal result (`200`).

Calls on one model are serialized. Two CPU workers serve independent models,
with at most 66 admitted calls across execution and waiting. The server retains
up to 1,000 models; delete unused ones to release capacity. Bodies are limited to
2 MiB (`413` on overflow).

Operations have a default deadline of 30 seconds, including admission waiting.
Decision `/recommend`, `/evaluate`, and `/propose` accept `timeoutMillis` from
1 to 300,000. Cooperative deadlines return `408`; disconnected requests cancel
active searches. A deadline or disconnect does not roll back committed updates;
inspect model state before retrying feedback.

State lives in memory and is lost on restart. The server has no authentication
layer and binds to loopback by default. The earlier experimental `/v1/problems`,
`/v1/optimizers`, and `/v1/stats` collections are replaced by `/v1/models`.
