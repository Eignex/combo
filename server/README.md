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
      "expr": {"add":["x",1.0]},
      "inner": {"ewma":{"alpha":0.1}}
    }
  }
}
```

The same observation then produces a mean of `43.0`. Here `"x"` is the input
reference in the canonical expression format; expressions are structured JSON.
The server passes `stat` directly to the native serializable specification.
Filters, weighting, groups, and other compositions use that same format.

## Decisions with a known objective

Declare decision variables and context supplied at request time. An objective
names the variable to minimize or maximize:

```http
POST /v1/models
Content-Type: application/json

{
  "variables": {
    "workers": {"type":"integer","min":1,"max":8},
    "load": {"type":"integer","role":"context","min":1,"max":8}
  },
  "constraints": [
    {"terms":{"workers":1,"load":-1},"atLeast":0}
  ],
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
an integer `cost` and adding `{"terms":{"cost":1,"workers":-2},"equal":0}`
allows `{"minimize":"cost"}`. There is no expression string language.

Without an objective, `/recommend` returns any feasible assignment. Search results
distinguish `feasible`, `optimal`, `best_found`, `infeasible`, `unbounded`, and
`unknown`. Real-valued optimization conservatively reports `best_found` rather
than claiming a proof of continuous optimality.

## Decisions with an observed objective

When an objective must be measured, give its variable the role `observed`:

```http
POST /v1/models
Content-Type: application/json

{
  "variables": {
    "workers": {"type":"integer","min":1,"max":8},
    "latency": {"role":"observed"}
  },
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

## Initial scope and operation limits

Decision variables support bounded integers and reals, Booleans, and categorical
labels. Context supports bounded integers and Booleans. Constraints currently
support integer coefficients on integer and Boolean variables, with exactly one
of `atMost`, `atLeast`, or `equal`. An observed objective is one real outcome;
constraints cannot depend on it. Attached outcome statistics must accept scalar
observations. Model definitions use ordinary serializable data classes, and
statistic specifications use their existing canonical JSON without translation.

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
