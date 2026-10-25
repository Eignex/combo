# Eignex Server

An API-only Ktor application that calls Klause, Kumulant, and Combo in the same JVM.
The `:server` application and `:combo` library are separate Gradle modules in the same
repository. The application uses the `com.eignex.server` package; the library keeps
its `com.eignex.combo` packages and Maven coordinates. Ktor belongs to the server module.

```sh
./gradlew :server:run
```

The default address is `127.0.0.1:8080`. Set `EIGNEX_SERVER_HOST` and `EIGNEX_SERVER_PORT` to change it.
The application uses JDK 25, provisioned by Gradle. `./gradlew :server:installDist`
builds a distribution under `server/build/install/eignex-server`; its `bin/eignex-server`
launcher also needs JDK 25.

## Job API

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/health` | Process health |
| POST | `/v1/jobs` | Submit a job; returns `202` and a `Location` header |
| GET | `/v1/jobs` | List retained job snapshots in submission order |
| GET | `/v1/jobs/{id}` | Poll status and retrieve the result |
| POST | `/v1/jobs/{id}/cancel` | Cancel queued or running work; repeat calls are safe |
| DELETE | `/v1/jobs/{id}` | Delete a terminal job record; active jobs return `409` |

Submission wraps a typed `request` and an optional `timeoutMillis` (default 30,000;
range 1–300,000). A snapshot contains `id`, `operation`, `status`, `submittedAt`, and,
when available, `startedAt`, `finishedAt`, `result`, and `error`.

Statuses are `queued`, `running`, `succeeded`, `failed`, `cancelled`, and `timed_out`.
Solver outcomes such as `unsat` or `unknown` are successful computations: inspect
`result.verdict` rather than treating job success as a satisfiability claim.

Malformed JSON, unknown operations or fields, and invalid request bounds return `400`.
Model compilation and observation validation happen on the worker; failures appear
in the job snapshot with `error.code = "invalid_job"`. Unexpected worker failures
use `execution_failed` and are logged on the server. Missing jobs return `404`.

## Kumulant summaries

```sh
curl -s http://127.0.0.1:8080/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{"request":{"type":"kumulant.summary","values":[1,2,3,4]}}'
```

Poll the returned `Location`. The result is
`{"count":4,"mean":2.5,"variance":1.25,"min":1.0,"max":4.0}`.
Variance is the population variance. Inputs contain 1–100,000 finite values.

## Klause solving

```sh
curl -s http://127.0.0.1:8080/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "request": {
      "type": "klause.solve",
      "schema": {
        "entries": {
          "enabled": {"$type":"bool"},
          "workers": {"$type":"int","min":1,"max":8},
          "strategy": {"$type":"nominal","labels":["fast","thorough"]}
        }
      },
      "maxDecisions": 100000,
      "seed": 42
    }
  }'
```

The schema uses Klause's existing `SchemaDef` wire format, including named
constraints. Nested schema types use `$type`; the job operation uses `type`.
The initial API supports Boolean, integer, nominal, float, and presence variables.
Set variables and file-format ingestion are not exposed by this operation.

Results have `verdict` set to `sat`, `unsat`, or `unknown`. A `sat` result contains
`values` keyed by variable name; `unknown` includes a `reason` from Klause.
`maxDecisions` defaults to 100,000 and accepts 1–10,000,000.

## Combo suggestions

```sh
curl -s http://127.0.0.1:8080/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "request": {
      "type": "combo.suggest",
      "space": {
        "variables": {"workers":{"$type":"int","min":1,"max":8}}
      },
      "observations": [
        {"values":{"workers":1},"reward":0.2},
        {"values":{"workers":4},"reward":0.8}
      ],
      "count": 2,
      "seed": 42,
      "maximize": true
    }
  }'
```

This operation fits Combo's forest/EI optimizer to the supplied history and returns
up to `count` feasible configurations in `result.suggestions`. `result.termination`
reports whether the requested count was reached or the feasible search stopped early.
Before five observations, the optimizer uses its initial random design.

`space` is Combo's existing `DecisionSpaceDef`, with constraints, nested and optional
spaces, scale metadata, and Boolean/integer context. Supply required context by name
in `request.context`; omitted optional context stays absent. Each observation can
also specify its own `context` and positive `weight` (default 1).

Observations are checked for feasibility before training. Their `values` must include
every flattened variable, including context and presence gates; returned suggestions
use that same shape. Observation import currently supports finite Boolean, integer,
and nominal spaces. Float spaces accept suggestion requests without observations.
Limits are 64 suggestions and 10,000 observations per job.

Each request creates an independent optimizer. Suggestions in one batch exclude each
other; reservations and learner state are not shared across jobs. A client keeps its
history and sends it with the next request. Stateful studies and streaming model
updates are future API extensions.

## Execution and lifecycle

Two CPU workers execute jobs independently of Ktor request handling, with a queue of
64 jobs and at most 1,000 retained records. Saturation returns `503` with `Retry-After`;
delete terminal records to free retained capacity. Request bodies are limited to 2 MiB
(`413` on overflow). Completed jobs release their input payloads and retain only the
snapshot and result.

The execution timeout starts when a worker starts the job. Cancellation and deadlines
are cooperative: the adapters check between work units and pass cancellation into
Klause searches. Cancellation immediately closes the API record and discards late
results; a running library call releases its worker when it observes the cancellation.
Server shutdown requests cancellation of all outstanding work.

Job records and results live in memory and are lost on restart. This initial server
has no authentication layer and binds to loopback by default.

Run the HTTP and worker lifecycle tests with `./gradlew :server:test`. They use Ktor's
in-process test host and real library adapters; no external service is required.
