# Eignex Server

An API-only Ktor application for repeated interaction with Klause, Kumulant, and
Combo. Create a resource once, then query or update its retained state by identifier:

| Resource | Retained state | Repeated operations |
| --- | --- | --- |
| `/v1/problems` | Compiled Klause schema, baked problem, and solver session | Solve or sample under different assumptions |
| `/v1/statistics` | Kumulant accumulator and observation count | Update, read, merge, reset |
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

For each collection, `POST` creates a resource and returns `201`, an `{id, type}`
response, and a `Location` header. `GET` lists resource identifiers, `GET /{id}`
inspects one resource, and `DELETE /{id}` releases its state. Operations return their
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

## Kumulant statistics

Create an accumulator, then send only new observations:

```http
POST /v1/statistics
Content-Type: application/json

{"type":"summary"}
```

```http
POST /v1/statistics/{id}/updates
Content-Type: application/json

{"values":[1,2]}
```

A subsequent update with `{"values":[3,4]}` returns
`{"count":4,"mean":2.5,"variance":1.25,"min":1.0,"max":4.0}`.
`GET /v1/statistics/{id}/snapshot` returns that same evidence without changing it.
Variance is population variance. The summary accumulator retains sufficient
statistics rather than storing or replaying the input stream.

Post a snapshot to `/v1/statistics/{id}/merge` to combine independent partial results.
The source accumulator remains unchanged. `POST /v1/statistics/{id}/reset` clears the
recipient. A snapshot with zero observations has zero-valued fields.

The initial accumulator type is `summary`. Update batches accept 1–100,000 finite
values. Updates and merges are validated before replacing the accumulator's state;
overflow or invalid input leaves its previous evidence intact. Counts are limited to
2^53 observations, matching the exact integral weight range of the accumulator.

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
