# `_mp2_energy_kernel` profile investigation — findings (#33)

**Date:** 2026-04-14.
**Tracker:** [#33](https://github.com/trnsci/trnblas/issues/33).
**Context:** [#15](https://github.com/trnsci/trnblas/issues/15) M2
shipped a fused energy kernel at 1.48× vs torch.
[#31](https://github.com/trnsci/trnblas/issues/31)'s first
optimization attempt ([PR #32](https://github.com/trnsci/trnblas/pull/32),
closed) showed no improvement. This doc records what we learned
when we tried to use `neuron-profile` to measure the bottleneck.

## TL;DR

**Capture works; extraction is blocked on the shipped DLAMI.** We
have an `ntrace.pb` for the `--fused-energy` medium bench on trn1,
but the `aws-neuronx-tools 2.29.18.0-d5fe7ba42` CLI can't convert
it to anything human-readable without either InfluxDB setup or a
tool/trace version alignment fix. Lacking a profiler, the perf
gap stays **unmeasured**. The next lever on #31 has to be a
kernel-shape change whose expected impact we can reason about
architecturally, not one we can attribute to a profile finding.

## What we tried

### A. `neuron-profile inspect` — CAPTURE WORKED

```
neuron-profile inspect -o <dir> /opt/aws_neuronx_venv_pytorch_*/bin/python \
    examples/df_mp2.py --bench --fused-energy --shape medium
```

Produced on the instance:

```
run-<timestamp>/
└── i-<instance>_pid_<pid>/
    └── <session-id>/
        ├── cpu_util.pb           (0 B — empty)
        ├── host_mem.pb           (0 B — empty)
        ├── ntrace.pb             (3.9 MB — device trace)
        └── trace_info.pb         (5 KB — metadata)
```

**Profiler overhead was small on the kernel under test.**
Unprofiled warm-medium energy step: 5.43 s. Profiled: 5.77 s
(+6%). Profiled wall time on other DF-MP2 steps (chol, half,
metric) blew up 10× — fine for our purposes since we're only
interested in the fused energy kernel.

### B. `neuron-profile show-session` — TRACE-FORMAT REJECTION

```
$ neuron-profile show-session -s ntrace.pb
NTFF version 130 is not supported in this version of the tool
(supported: 1 - 6). Please upgrade aws-neuronx-tools.
```

The capture tool and the `show-session` tool ship in the **same**
package (`aws-neuronx-tools 2.29.18.0-d5fe7ba42`), and the capturer
produces a format the shipped `show-session` can't read. Looks like
an internal packaging/versioning mismatch in 2.29; plausibly fixed
in a future release.

### C. `neuron-profile view --disable-ui --ingest-only` — INFLUXDB REQUIRED

```
$ neuron-profile view --disable-ui --ingest-only -d <session> --data-path <dir>
influxdb not setup correctly: exec: "influx": executable file
not found in $PATH
```

`view` is a web-UI-backed ingest tool, and "ingest" means "push
into an InfluxDB instance." The DLAMI doesn't pre-install InfluxDB
and setting it up (install, configure, wire neuron-profile at it)
is its own infra project.

### D. `neuron-top -b` / `neuron-monitor` — LIMITED SIGNAL

- `neuron-top -b` (batch mode) doesn't exist — neuron-top is purely
  interactive.
- `neuron-monitor` produces a JSON snapshot of runtime state. With
  no workload running it shows `neuron_runtime_data: []`. Running
  it concurrently with a bench would give a coarse per-core
  utilization sample, but not the per-op trace we actually want.

## What we can say with confidence

Even without the profiler, the measurements we already have
falsify some hypotheses and point at others:

1. **HBM bandwidth is not the bottleneck.** Large warm energy is
   ~30 s/chunk; HBM floor for 33 GB of intermediate traffic at
   ~700 GB/s is ~47 ms. Measured is ~600× over floor.

2. **Raw tensor-core compute is not the bottleneck.** The upstream
   GEMM is ~0.5 s at 35 TFLOPs; the ~30 s remainder is not compute.

3. **Per-op dispatch overhead in denom construction is not the
   bottleneck.** PR [#32](https://github.com/trnsci/trnblas/pull/32)
   collapsed the denom build-up from 5 ops per `(i, j, s)` to 1 op
   + pre-loop hoist. Result: no measurable change (1.48× → 1.50×
   at medium, 1.47× → 1.49× at large — both within noise). The
   NEFF compiler is already doing this work on our behalf.

4. **The speedup ratio is shape-invariant** (1.48× medium vs 1.47×
   large). Per-pair launch cost scales with `nocc²` the same way
   torch does. Whatever the actual bottleneck is, it scales with
   the pair loop.

## The open hypothesis

Without a profile trace we can't confirm this, but the consistent
~1.48× ratio and the independence from dispatch-count are
consistent with **a cross-pair synchronization fence in the
kernel**:

- Our `_mp2_energy_kernel` stores `e_partial[P_TILE, i, j]` at the
  end of each `(i, j)` pair — an HBM write.
- The compiler probably inserts dependencies so each pair's
  computation sits behind the previous pair's store completing.
- There's no structural reason the compiler would cross-pair
  pipeline: every pair's Vector-Engine work writes into the same
  HBM tensor, and without explicit relaxation from us the
  compiler has to serialize.

If that's the bottleneck, the lever is **cross-pair batching** —
accumulate many pairs' partials in SBUF before one large HBM
store. That's a kernel-shape change, not an op-level optimization,
and it's the natural next-issue for #31.

## What this doesn't tell us

- Actual per-engine utilization. Vector Engine may be at 20% or
  80% — different optimizations apply.
- Whether the Tensor Engine is truly idle during the reduction
  (RFC expectation) or inadvertently scheduled by the compiler.
- Memory-subsystem time (HBM read latency, PSUM/SBUF stalls) vs
  compute time.
- Any compiler-inserted ops we didn't write.

These remain unknowns until a working profile extraction path is
available.

## Recommended next steps

1. **Don't invest further in profiler tooling on 2.29.** Revisit
   when `aws-neuronx-tools` ships a fix for the NTFF version
   mismatch **or** when there's a compelling reason to set up
   InfluxDB on the CI DLAMI (e.g. multiple perf investigations
   queue up and justify the infra work).

2. **File a cross-pair batching issue as the concrete next step
   for #31.** Scope: accumulate `K` pairs' partials in SBUF per
   HBM store (K picked at trace time from SBUF budget). Expected
   impact isn't measured, but the hypothesis is explicit and the
   change is a bounded kernel-shape rework.

3. **Keep `scripts/run_neuron_profile.sh` checked in.** The
   capture path works and the doc above explains the extraction
   constraint. Future invocations are one script edit away when
   extraction unblocks.
