# combo-bench

Kotlin/JVM synthetic optimization benchmarks for
[eignex-lab](https://github.com/Eignex/eignex-lab). The benchmark module owns its
workloads and measurements; the lab owns revision builds, execution, comparisons,
and retained artifacts. No benchmark results or Python tooling are stored here.
Historical report-driven experiment mains have been replaced by this suite;
external SMAC/BoTorch references and archived-fixture audits are no longer included.

Use the authenticated HTTP client at `https://lab.eignex.com`; see the lab's
[run guide](https://github.com/Eignex/eignex-lab/blob/main/docs/runs.md).
SSH access, host discovery, and server administration belong to agents with an
explicit eignex-lab operations task. Report operational dependencies to that owner.

## Build and smoke check

The installed launcher requires JDK 25; set `JAVA_HOME` to that JDK if your shell
uses an older Java. Gradle provisions its own JDK 25 toolchain for builds and tests.

```sh
./gradlew :combo-bench:test :combo-bench:installDist
BENCH=combo-bench/build/install/combo-bench/bin/combo-bench
"$BENCH" describe
mkdir -p combo-bench/build/smoke
printf '%s\n' '{"parameters":{"smoke":"true"}}' > combo-bench/build/smoke/request.json
"$BENCH" plan combo-bench/build/smoke/request.json > combo-bench/build/smoke/plan.json
jq '.cases[0]' combo-bench/build/smoke/plan.json > combo-bench/build/smoke/case.json
"$BENCH" run combo-bench/build/smoke/case.json combo-bench/build/smoke/out
```

JDK 25 and the Vector API/native-access flags are configured in both the installed
launcher and [lab manifest](lab/bench.json). The `com.eignex:lab-bench:0.1.0` SDK
stays in this unpublished application module. Like Kumulant and Koblas, it resolves
from public GitHub Release assets. SDK development can override `labBenchVersion`
and `labBenchRepository`; normal runs need neither credentials nor a sibling lab build.

## Lab submission

Save a request under an ignored build directory and submit it with the maintained
client's `lab run REQUEST.json` command:

```json
{
  "name": "combo-optimization",
  "repository": "Eignex/combo",
  "base": {"ref": "COMMIT_SHA"},
  "bench": {
    "manifest": "combo-bench/lab/bench.json",
    "request": {"parameters": {
      "workloads": "quadratic,multimodal,logarithmic,noisy,conditional",
      "algorithms": "gp,forest,random",
      "seeds": "3,11,19,29,43,61",
      "repetitions": "3",
      "budget": "32"
    }}
  }
}
```

Each case contains every selected algorithm for one workload, seed, and repetition.
For revision A/B jobs, all revision arms and algorithms for that row run on the same
machine in one worker attempt. Other rows, preparation, and replacement attempts
may use other compatible machines. Use automatic placement without pinning the
whole job. Worker loss replaces the complete unfinished row; never combine arms
across machines or attempts. Retain failed-attempt and preparation costs, and diagnose
repeated infrastructure losses separately. Algorithm order rotates across repetitions;
repetitions reuse the seed and measure replay/runtime variation, not independent seeds.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `workloads` | All five above | Comma-separated workload names |
| `algorithms` | `gp,forest,random` | All arms retained in each row |
| `seeds` | `3,11,19,29,43,61` | Distinct integer optimizer/noise seeds |
| `repetitions` | `1` | Repeated complete paired rows |
| `budget` | `32` | External evaluations per algorithm |
| `smoke` | `false` | Defaults to quadratic, seed 3, four evaluations |
| `memory-mib` | `2048` | Case memory; minimum 2048 for the 1 GiB heap |
| `timeout-seconds` | `600` | Whole case deadline, enforced by the lab |

Explicit parameters override smoke defaults. Unknown parameters, duplicate selections,
and invalid resource settings fail during planning. Execution revalidates frozen cases.

## Workloads and measurements

Every objective is minimized and has known optimum 1:

- `quadratic`: `1 + (x - .73)^2` on `[0,1]`.
- `multimodal`: `1 + .2(x - .31)^2 + .1 sin(6π(x - .31))^2` on `[0,1]`.
- `logarithmic`: `1 + ((ln(x) - ln(1e-6))/ln(1e6) - .62)^2` on `[1e-6,1]` with log geometry.
- `noisy`: `1 + (x - .31)^2 + 3(y - .72)^2` on `[0,1]^2`, with independent uniform
  observation noise in `[-.05,.05]`. Its noise RNG restarts identically for each arm;
  the optimizer receives the known observation variance `.0025/3`.
- `conditional`: a continuous quadratic around `.62`, a `slow`/`fast` category with
  `.1` slow penalty, and an optional integer child in `[1,6]` with `.02(size - 4)^2`
  active penalty or `.3` inactive penalty.

GP and forest use normalized search coordinates, standardized responses, EI, the
same budget-aware Latin hypercube, 20% interleaved feasible exploration, and local
search with 64 candidates, four starts, four steps, and eight neighbors. Forests
use randomized splits across all features. `random` uses the same initial design
and feasible witness machinery with exploration probability 1 and performs no
model fits. Completed points are excluded for every arm. Witness generation is
heuristic and does not promise uniform constrained sampling.

Each case writes `result.json` and retains `report/{gp,forest,random}.csv` for the
selected arms plus runtime/configuration metadata. Traces record observed and
noiseless values, cumulative external cost (one per evaluation), best evaluated
regret, model fits, acquisition evaluations, and normalized coordinates. Generic
metrics expose final regret and work counts. Truth is used only for diagnostics;
it never supplies optimizer labels or suggestions. Incomplete budgets fail the case.

These comparisons measure sample efficiency. Model fitting, search, and runtime
overhead are separate from the external evaluation budget. Elapsed seconds include
the optimization loop and trace writing, exclude model construction, and have no
warmup or JMH controls; do not treat them as a throughput benchmark. Small synthetic
workloads and a few seeds establish no universal ranking. Tests check the contract,
finite complete traces, and replay, without asserting stochastic superiority.

Keep generated output under `combo-bench/build/` for local checks; use lab artifacts
for experiments and comparisons. Do not copy evidence or results into source folders.
