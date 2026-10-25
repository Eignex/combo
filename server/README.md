# Eignex Server

An API-only Ktor application for repeated interaction with Klause, Kumulant, and
Combo. Create a resource once, then query or update its retained state by identifier:

| Resource | Retained state | Repeated operations |
| --- | --- | --- |
| `/v1/problems` | Compiled Klause schema, baked problem, and solver session | Solve or sample under different assumptions |
| `/v1/stats` | Native Kumulant stat or bandit | Update, read, choose, reset |
| `/v1/optimizers` | Combo decision space, optimizer, history, and trial reservations | Suggest, complete, fail, cancel |

The `:server` application and `:combo` library are separate Gradle modules in the same
repository. The application uses `com.eignex.server`; the library keeps its
`com.eignex.combo` packages and Maven coordinates. Ktor belongs to the server module.

```sh
./gradlew :server:run
```

The default address is `127.0.0.1:8080`. Set `EIGNEX_SERVER_HOST` and
`EIGNEX_SERVER_PORT` to change it. Gradle provisions JDK 25 for development.
`./gradlew :server:installDist` creates `server/build/install/eignex-server`; its
`bin/eignex-server` launcher also needs JDK 25.

## Shared lifecycle

For each collection, `POST` creates a resource and returns `201`, an identifier,
and a `Location` header. Stats creation returns `{id}`; problems and optimizers
also return `type`. `GET` lists resource identifiers, `GET /{id}` reads one resource,
and `DELETE /{id}` releases its state. Operations return their
results directly. `GET /health` reports process health.

Malformed input, unknown fields, invalid models, and invalid operation parameters
return `400`. Unknown resources and trials return `404`; completing or changing a
closed trial returns `409`. Saturation returns `503` with `Retry-After`.
Errors have the shape `{code, message}`.

## Klause problems

Create a problem with the existing Klause `SchemaDef` wire format:

```http
POST /v1/problems
Content-Type: application/json

{
  "schema": {
    "entries": {
      "enabled": {"$type":"bool"},
      "workers": {"$type":"int","min":1,"max":8},
      "strategy": {"$type":"nominal","labels":["fast","thorough"]}
    }
  }
}
```

Then use its identifier for repeated queries, without resending or recompiling the schema:

```http
POST /v1/problems/{id}/solve
Content-Type: application/json

{"assumptions":{"enabled":true,"workers":4},"maxDecisions":100000,"seed":42}
```

`POST /v1/problems/{id}/sample` accepts the same request. Assumptions belong to one
call and do not carry into the next. The retained session reuses the compiled and
baked problem; calls start fresh searches. The schema is immutable for the resource's
lifetime. Create a replacement resource to use a different schema.

Results contain `verdict` (`sat`, `unsat`, or `unknown`); `sat` includes named `values`,
and `unknown` includes the solver's `reason`. An infeasibility proof is a normal
solver result and returns `200`.

The initial API supports Boolean, integer, nominal, float, and presence variables,
plus named constraints. Query assumptions support Boolean, integer, and nominal
variables. Float assumptions, set variables, and file-format ingestion are not
exposed. `maxDecisions` defaults to 100,000 and accepts 1–10,000,000.

## Kumulant stats and bandits

`POST /v1/stats` accepts Kumulant's canonical JSON directly: any `StatSpec`,
`UnivariateBanditSpec`, or `ContextualBanditSpec`. The native spec selects the model;
there is no extra mode field or server expression language.

Create a simple exponentially weighted moving average:

```http
POST /v1/stats
Content-Type: application/json

{"ewma":{"alpha":0.1}}
```

The response is `{"id":"..."}`, with `Location: /v1/stats/{id}`. Feed it one observation:

```http
POST /v1/stats/{id}/update
Content-Type: application/json

{"value":42.0}
```

The first update returns `{"totalWeights":1.0,"mean":42.0}`. Updates return the
current native snapshot, and `GET /v1/stats/{id}` reads it without changing it.

To add one to every observation automatically, use the native composition:

```json
{
  "transform": {
    "expr": {"add":["x",1.0]},
    "inner": {"ewma":{"alpha":0.1}}
  }
}
```

The same `{"value":42.0}` update then returns a mean of `43.0`. Filters, weighting,
feedback, groups, and other native decorators use this same Kumulant format.
Grouped snapshots retain their native `results` structure.

### Observation shapes

Send one observation per request. Fields depend on the model's input modality:

| Model | Update body |
| --- | --- |
| Series stat | `{"value":42.0}` |
| Discrete stat | `{"value":42}` |
| Paired stat | `{"x":1.0,"y":2.0}` |
| Vector stat | `{"x":[1.0,2.0]}` |
| Regression stat | `{"x":[1.0,2.0],"y":3.0}` |
| Ordinary bandit | `{"arm":0,"reward":0.8}` |
| Contextual bandit | `{"arm":0,"x":[1.0,2.0],"reward":0.8}` |

All updates accept an optional finite `weight`, defaulting to `1.0`. Zero weights
and negative weights follow Kumulant's native semantics. Stat updates also accept
`timestampNanos`; otherwise the server uses its monotonic clock. Time-dependent
reads can specify `GET /v1/stats/{id}?timestampNanos=...` using that same clock domain.

Numeric inputs must be finite JSON numbers, and discrete observations and timestamps
must be integers. Unknown fields, missing inputs, and incorrect context dimensions
return `400` before the observation is applied. Native model validation still applies.
Undefined or non-finite numbers in statistical results are returned as JSON `null`.

### Bandits

Create an ordinary bandit using a native spec:

```json
{"multiArmed":{"nbrArms":2,"policy":{"greedy":{}}}}
```

`POST /v1/stats/{id}/choose` with `{}` returns `{"arm":0}`. Report a reward through
`/update`; subsequent choices use the retained evidence. Set `?seed=42` on creation
to choose a reproducible random source; the default seed is `0`.

For a contextual bandit, creation can use:

```json
{
  "regressionContextual": {
    "nbrArms":2,
    "regression":{"bayesianRegression":{"featureSize":2}},
    "posterior":{"linUcb":{}},
    "exploration":1.0
  }
}
```

Choose with `{"x":[1.0,2.0]}` and report the same context with the resulting arm
and reward. Bandit reads and updates return an `arms` array of native snapshots.
Pooled regressors also include `global`; tracked bandits include their `inner`
snapshot and configured observation statistics. Decorators retain Kumulant's
native update and scoring semantics.

`POST /v1/stats/{id}/reset` resets a stat or bandit to its configured baseline and
returns the snapshot. `DELETE /v1/stats/{id}` releases it. `GET /v1/stats` lists
identifiers. The experimental summary-only `/v1/statistics` routes are replaced by
this collection.

## Combo optimizers

Create the optimizer and decision space once:

```http
POST /v1/optimizers
Content-Type: application/json

{
  "space": {"variables":{"workers":{"$type":"int","min":1,"max":8}}},
  "seed": 42,
  "maximize": true
}
```

Request suggestions on that optimizer:

```http
POST /v1/optimizers/{id}/suggestions
Content-Type: application/json

{"count":2,"context":{}}
```

The response contains `termination` and `trials`. Each trial has an `id`, `status`,
and named `values`. Reservations remain pending across subsequent HTTP requests,
so concurrent evaluation batches cannot reserve the same configuration twice.
`GET /v1/optimizers/{id}/trials` lists issued trials and their current state.

Report feedback using the returned trial identifier:

```http
POST /v1/optimizers/{id}/trials/{trialId}/complete
Content-Type: application/json

{"reward":0.8,"weight":1.0}
```

The optimizer records the observation once and uses it in later suggestion rounds.
Feedback can arrive out of order. The server retains the original sample and context,
so completing a float-valued trial preserves its checked assignment without asking
the client to reconstruct it. Optional `noiseVariance` must be finite and nonnegative.

`POST /v1/optimizers/{id}/trials/{trialId}/fail` takes `{"reason":"..."}`;
`POST /v1/optimizers/{id}/trials/{trialId}/cancel` takes no body. Both release the
configuration without adding training evidence. A closed trial rejects further
feedback or state changes. Inspecting the optimizer reports observation and pending
trial counts.

The initial optimizer is forest/EI. Before five completed observations it uses its
initial random design. The decision space accepts constraints, nested and optional
spaces, scale metadata, and Boolean/integer context. Supply required context by name
on each suggestion call; omitted optional context stays absent. Up to 64 trials can
be requested per call. If search stops early, inspect `termination` and the returned
partial batch.

## Concurrency and storage

Calls on one resource are serialized; independent resources can execute concurrently.
Two CPU workers handle library operations, with at most 66 admitted calls across
execution and waiting. Each collection retains at most 1,000 resources. Delete unused
resources to release capacity. Request bodies are limited to 2 MiB (`413` on overflow).

The default operation deadline is 30 seconds, including time waiting for admission to
its resource. Problem creation, solver queries, optimizer creation, and suggestions
accept `timeoutMillis` in 1–300,000. Cooperative deadlines return `408`; request
cancellation is passed into active searches. Library work stops when it observes
cancellation. A deadline or disconnected client does not roll back an update already
committed; inspect resource or trial state before retrying feedback.

Resource state lives in memory and is lost on restart. The initial server has no
authentication layer and binds to loopback by default.

Run the HTTP and resource lifecycle tests with `./gradlew :server:test`. They use
Ktor's in-process test host and real library adapters, without external services.
