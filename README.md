# berth — inference placement primitives

A [Reckon Research](https://www.reckonresearch.com/berth) project. Docs: [docs.reckonresearch.com](https://docs.reckonresearch.com).

Predict what a workload costs and how fast it runs on a given accelerator,
before you rent it.

```bash
pip install berth-placement
berth place --workload-class voice --model llama3-8b --incumbent h100-pcie --slo-ms 800
```

[Docs](https://docs.reckonresearch.com) ·
[The Placement Index](https://reckonresearch.com/index/) ·
[Defect register](https://github.com/reckonresearch/berth/blob/main/DEFECTS.md)

[![ci](https://github.com/reckonresearch/berth/actions/workflows/ci.yml/badge.svg)](https://github.com/reckonresearch/berth/actions)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/reckonresearch/berth/blob/main/LICENSE)

**Where should this workload run?** berth answers that with four primitives —
`profile / estimate / place / migrate` — over an auditable roofline model,
trace-based calibration with confidence intervals, and tail-aware fleet
sizing. Every estimate is annotated with its **placement premium**: the
measured cost of running on the wrong silicon.

It is not a black box that makes placement "easy". It is a clean abstraction
that shields you from orchestration while preserving control of the policy —
your objective and constraints are plain Python.

**Status: validated against measured hardware on two cells.** P0 ran
Llama-3-8B under vLLM on a rented NVIDIA L40S and H100 PCIe, 60 traces each,
and scored the estimator against what the hardware did. Using untuned
spec-sheet priors, decode latency predicts to 10.6% (L40S) and 4.2% (H100
PCIe) against a 15% gate published before the first run. Everything else in
the fleet is a spec-sheet prior and is labelled `prior` on every line the CLI
prints. The market backend is a seeded simulator; a production backend
implements the 4-method `Backend` protocol.

Full results, including two unresolved discrepancies and three instrument
defects that all understated the model:
[Validation (P0)](https://docs.reckonresearch.com/validation-p0/).

## Install

```bash
pip install berth-placement          # the CLI command is berth
```

The distribution is `berth-placement`; the command it installs is `berth`. The
short name on PyPI belongs to an unrelated project.

A first number needs no Python:

```bash
berth estimate --model llama3-8b --batch 32
berth premium  --model llama3-8b --prices l40s=0.99 h100-pcie=3.35
berth list                            # fleet and models, each tagged MEASURED or prior
```

From source, to run the tests and the measurement harness:

```bash
git clone https://github.com/ReckonResearch/berth && cd berth
pip install -e .            # stdlib-only core, no dependencies
python -m pytest tests/ -q  # 61 tests
```

## Quickstart

```python
from berth import MODELS, PlacementClient, PlacementPolicy, SimBackend, WorkloadSpec, min_cost

client = PlacementClient(SimBackend(seed=0))
sig = client.profile(WorkloadSpec(model=MODELS["llama3-70b"], target_batch=16,
                                  p99_ttft_ms=500.0, arrival_rps=40.0))
for e in client.estimate(sig):                       # premium-annotated fleet table
    if e.feasible:
        print(e.silicon, f"${e.cost_per_mtok:.2f}/Mtok", f"premium {e.placement_premium:.0%}")

handle = client.place(sig, PlacementPolicy(          # policy = plain Python
    objective=min_cost,
    constraints=(lambda e: e.ttft_ms < 500,),
))
handle = client.migrate(handle)                      # re-place on market drift
```

Tutorials (each runnable, with expected output): [101 hello](tutorials/101_hello_berth.py) ·
[201 policies & SLOs](tutorials/201_policies_and_slos.py) ·
[301 calibration](tutorials/301_calibrate_from_traces.py) ·
[401 tail-aware sizing](tutorials/401_tail_aware_sizing.py).
Docs: [overview](docs/index.md) · [concepts](docs/concepts.md) · [hardware runbook](docs/bench.md).

## Where berth sits

berth decides **where**; Dynamo/vLLM/SGLang run it there; SkyPilot/Kubernetes
deploy it there. Upstream benchmarks (e.g. InferenceX) measure what hardware
can do on canonical workloads; berth turns measurements into placement
decisions for *your* workload, SLO, and prices.

## Scope

**Now (H1):** placing inference workloads onto the right silicon — across GPU
SKUs and across providers — under cost/latency policies you write. The
placement premium (cost vs best feasible placement) is measurable today and
is what this repo estimates, calibrates, and captures.

**Roadmap (H2):** heterogeneous multi-silicon routing (ASIC/FPGA/CPU targets
alongside GPUs). The estimator's silicon model is already
architecture-agnostic; what H2 waits on is calibration data and rentable
inventory for non-GPU targets, not new abstractions.

**Research (H3):** placement-aware interconnect and scheduling below the
software layer. Long-horizon; H1's measured premium data is the evidence
base that justifies (or kills) it.

**The index layer (spans all horizons):** every measured trace recalibrates
the fleet model, and the calibrated placement-premium index — with published
confidence intervals and raw traces — compounds across horizons. Routing
decisions expire in seconds; the measurement corpus does not. Premium
durability note: per-SKU premiums compress as software stacks mature, but
each new silicon generation ships with immature software and re-opens them,
and cross-provider price/queue spreads persist regardless — the premium is
a renewable resource driven by hardware release cadence, not a one-time gap.

Non-goals: berth is not an inference engine and not a model server — it
decides *where* engines run, and stays neutral about *which* engine.
Batch/training-job placement is deliberately out of scope for now: it is
technically easier (no latency SLO, interruptible) but already served by
mature open-source schedulers; latency-SLO inference is where placement is
both hard and unserved.

## The four primitives

[`DEFECTS.md`](https://github.com/reckonresearch/berth/blob/main/DEFECTS.md) is the register: eleven instrument failures, what
caused each, how it was caught, and the test that fails if it returns.

client = PlacementClient(SimBackend(seed=42))

sig = client.profile(WorkloadSpec(model=MODELS["llama3-70b"], target_batch=16, p99_ttft_ms=500))
estimates = client.estimate(sig)                      # fleet-wide, premium-annotated
handle = client.place(sig, PlacementPolicy(
    objective=min_cost,                               # any Callable[[Estimate], float]
    constraints=(lambda e: e.ttft_ms < 500,),
))
handle = client.migrate(handle)                       # re-place on market drift, with hysteresis
```

Run `python demo.py` for the full loop; `python -m pytest tests/` for the suite.

## Model

Analytical roofline, deliberately auditable (every number derives from public
specs + two per-device efficiency factors):

- **Decode step** = max(compute time, memory time); batch amortizes weight
  reads, KV reads scale per sequence. Reports which roof binds.
- **Prefill** = compute-bound FLOPs / effective FLOPS → TTFT.
- **MoE**: memory footprint from total params, per-token cost from active
  params. These must never be conflated.
- **TP**: minimum device count that fits weights + KV in 0.9× aggregate
  memory; scaling efficiency decays per doubling.
- **placement premium** = cost_per_Mtok / best_feasible − 1, annotated per estimate — the measured cost of running a workload on the wrong silicon.

## Architecture decisions

- `Backend` is a 4-method protocol (`fleet/price_hr/bind/release`).
  `SimBackend` is a seeded random-walk spot market; a production backend
  (real orchestrator or neocloud broker) swaps in without touching client or
  policy code. The protocol is the durable asset.
- Migration has hysteresis (`min_improvement`, default 15%) because real
  migrations cost warm KV caches and connection draining; a free-migration
  model churns itself to death.
- Estimator is analytical, not fitted. A measured-trace calibration layer
  belongs *on top* (adjusting `mfu`/`bw_eff` per silicon per workload class),
  keeping the schema neutral and inspectable.

## Calibration layer (v0.2)

`calibrate(prior_fleet, traces)` fits per-silicon `(mfu, bw_eff)` from
measured `TraceRecord`s by **direct roofline inversion** (TTFT -> mfu;
memory-bound TPOT -> bw_eff; compute-bound TPOT -> mfu), aggregated with
robust medians and iterated twice for bound reclassification. No optimizer
dependency; every fitted value traces to specific observations. Validation is
blind recovery: `generate_traces` runs a hidden true fleet + 5% lognormal
noise, and the fitter recovers all parameters within noise (holdout MAPE
10.8% -> 3.7%). Calibrated fleets plug into `SimBackend(fleet=...)` with zero
policy-code changes. Run `python demo_calibrate.py`.

## Queueing layer (v0.3)

Add `arrival_rps` to `WorkloadSpec` and estimates become fleet sizings: the
replica pool is an M/M/c queue on batch slots (c = replicas x batch, service
time = request residence = TTFT + out_tokens x TPOT). Erlang-C via the stable
Erlang-B recurrence gives P(wait); the M/M/c wait tail is exponential, so p99
TTFT = closed-form p99 wait + prefill service. `size_replicas` finds the
minimum fleet meeting the p99 SLO (or an 0.85-utilization convention when no
SLO is set), and `$ / Mtok` prices in the headroom: total fleet cost over
goodput at the offered arrival rate. Omitting `arrival_rps` preserves exact
v0.2 single-replica semantics. Run `python demo_queueing.py` — tightening the
TTFT SLO from 500ms to 300ms flips the cheapest silicon and knocks two
classes out entirely.

Known bias: residence times are modeled exponential; heavy-tailed output
lengths make true tails worse. The Allen-Cunneen SCV correction slots into
the calibration layer once measured residence variance exists.

## Per-class calibration + drift (v0.4)

`workload_class(sig)` buckets signatures by batch regime x context regime
(kernel-regime boundaries; conventions, revisit against per-cell variance).
`calibrate_classed` fits (mfu, bw_eff) per (silicon, class) with a strict
fallback hierarchy — class fit if the cell has >= min_traces, else the
per-silicon global fit, else the prior; thin cells never fabricate.
`ClassedFleet.resolver()` plugs into `PlacementClient(profile_resolver=...)`
so estimates use workload-appropriate profiles with zero policy-code changes.

A robust (median) global fit doesn't average a bimodal efficiency split — it
locks onto the majority mode and *hides* the minority class entirely. In the
demo, MI300X long-context bw_eff is 0.42 vs 0.60 elsewhere; the global fit
reads 0.590 and would misprice long-context work by ~40% with no indication.

`detect_drift` refits per time window and takes the OLS slope per parameter:
a maturing software stack appears as trending bw_eff (placement premium moving in
real time — index signal, not noise). Silent below min_per_window; no false
positives on stable silicon at the 0.05 threshold. Run `python demo_drift.py`.

## Production gap inventory (in order)

1. Real `Backend` against an orchestrator or neocloud broker (the protocol
   is 4 methods: `fleet / price_hr / bind / release`).
2. Trace ingestion from real serving metrics (vLLM/TGI/TRT-LLM exporters)
   replacing the synthetic generator.
3. Hardware calibration campaign: efficiency factors here are informed
   priors, not measurements. Predicted-vs-measured error on rented silicon
   is the gate for any accuracy claim.
4. Control plane: API service, authn/z, tenancy, persistence.
5. Real migration: connection draining, KV-cache handling, warm-up cost in
   the hysteresis term.

## Known limits / v1 targets

- Single-replica placement; no fleet-level bin-packing (ILP slot exists in
  `_select` when it's needed — YAGNI until then).
- No quantization-accuracy tradeoff axis (bytes_per_param is exposed; the
  accuracy dimension of the objective is not modeled).
- Class boundaries are fixed conventions; data-driven cell splitting (split
  where within-cell fit variance stays high) is the refinement.
- Drift slope is mildly attenuated by within-window averaging; fine for
  flagging, use endpoint window fits for magnitude.
- Prefill/decode disaggregation, chunked prefill, and speculative decoding
  are not modeled — they shift the roofline shape, beyond what scalar
  efficiency factors can absorb.
- p99 TPOT is still checked against the mean (prefill-interference under
  continuous batching is not modeled); p99 TTFT is now queueing-aware.

---

berth is built and maintained by Reckon Research, publisher of the
placement-premium index. Apache-2.0.

## Verify it, and tell us where it is wrong

berth ships `sounding`, the same harness that produced its own validation data.
Point it at a model server you already run, then score the estimator against
what your hardware actually did:

```bash
python -m bench.sounding --base-url http://localhost:8000 \
    --silicon h100-pcie --model llama3-8b --model-id meta-llama/Meta-Llama-3-8B \
    --out traces.jsonl
python -m bench.fit_overhead traces.jsonl            # fit your own prefill floor
python -m bench.validate     traces.jsonl --prefill-overhead-ms <fitted>
```

The prefill floor is a property of one (accelerator, driver, server, config)
tuple and is not predictable from a spec sheet: it measured 74.6 ms on the L40S
and 54.6 ms on the H100 PCIe. Profiles ship 0.0 and every run fits its own. Do
not carry someone else's.

Every record carries `source`, `measured` or `mock`, written by the harness.
`bench.validate` refuses a file mixing the two, and CI rejects any contribution
containing a mock record or a record without explicit provenance. A mock trace
and a hardware trace are otherwise indistinguishable on disk, and the corpus is
worth exactly as much as that distinction.

If berth misses on your silicon, open an issue with your `traces.jsonl`
attached. Disputes that arrive with traces outrank everything else, and
unfavorable results publish.
