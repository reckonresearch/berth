# berth — inference placement primitives

**Status: simulation-validated reference implementation. Not a hosted
service; nothing here executes inference on real hardware.** What is real:
the API contract (four primitives), the analytical roofline model, the
calibration and drift machinery, and the validation harness — all of which a
production backend implements against actual silicon by satisfying the
4-method `Backend` protocol.

berth applies the Tinker slice to inference placement: The user keeps the
policy loop (plain Python objective + constraints over `Estimate`); the
platform keeps execution (pricing, capacity, bind/release, migration).

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

```python
from berth import MODELS, PlacementClient, PlacementPolicy, SimBackend, WorkloadSpec, min_cost

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
