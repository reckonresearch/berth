# berth: a placement API for inference engineers

berth lets you focus on what matters in serving open-weight models — your
workload and your policy — while it handles the placement math: which
silicon, which provider, how many replicas, at what cost, under your SLO.

You write a simple policy loop that runs on your laptop, including your
objective and your constraints as plain Python. berth figures out what your
workload costs on every chip and provider it knows, doing the exact
computation you can audit by hand, and places it where your policy says.
To consider a new chip or a new provider, you only need one more row in
the fleet table.

berth gives you full control over the placement decision and all the
economic details. It is not a black box that makes placement "easy". It is
a clean abstraction that shields you from the complexity of the
cross-provider GPU market while preserving your control over what
"optimal" means for you.

## What you can do with berth

- **Estimate** any workload across the fleet in milliseconds: TTFT, TPOT,
  $/Mtok, which roofline binds, and the **placement premium** — the
  measured cost of running on the wrong silicon.
- **Size fleets against tail SLOs**: give an arrival rate and a p99 TTFT
  target; get replica counts with queueing headroom priced in. The
  cheapest chip at the mean is often not the cheapest at the tail.
- **Place and migrate** with policies you write: any objective, any
  constraints, hysteresis against market churn, forced moves on
  constraint violations.
- **Calibrate from your own traces**: closed-form inversion fits each
  chip's real efficiency from measured TTFT/TPOT, with bootstrap 95%
  confidence intervals and blind-recovery validation.
- **Detect software-stack drift**: a maturing runtime shows up as a
  trending efficiency factor — the premium moving in real time.

## Show me the code

```python
from berth import MODELS, PlacementClient, PlacementPolicy, SimBackend, WorkloadSpec, min_cost

client = PlacementClient(SimBackend(seed=0))

sig = client.profile(WorkloadSpec(
    model=MODELS["llama3-70b"],
    target_batch=16, avg_prompt_tokens=1024, avg_output_tokens=256,
    p99_ttft_ms=500.0, arrival_rps=40.0,
))

for e in client.estimate(sig):                 # premium-annotated fleet table
    if e.feasible:
        print(e.silicon, f"${e.cost_per_mtok:.2f}/Mtok",
              f"premium {e.placement_premium:.0%}")

handle = client.place(sig, PlacementPolicy(    # your policy, plain Python
    objective=min_cost,
    constraints=(lambda e: e.ttft_ms < 500,),
))
handle = client.migrate(handle)                # re-place on market drift
```

Changing the model is one string. Changing what "best" means is one lambda.

## Who it's for

Engineers who serve open-weight LLM/ASR models on GPUs they rent or
reserve, and who want the placement decision to be a measured, auditable
computation instead of a default. If your team knows its $/Mtok by heart —
or wants to — berth is for you.

## Under the hood

Every estimate is closed-form and derivable by hand: a roofline model with
quadratic-attention prefill, KV-pressure decode, MoE memory/compute
separation, and M/M/c queueing for tails. Fitted corrections layer on top
via trace calibration; they never replace the analytical core. Physics
changes ship with validator changes in the same commit — the term-by-term
checks in `bench/validate.py` are the spec. Read
[concepts](concepts.md) for the model and [the hardware runbook](bench.md)
for measuring your own silicon.

## Getting started

Work through the tutorials in order — each is runnable and states its
expected output:

1. [101 — hello berth](../tutorials/101_hello_berth.py): profile and
   estimate across the fleet
2. [201 — policies and SLOs](../tutorials/201_policies_and_slos.py):
   objectives, constraints, and when they disagree
3. [301 — calibrate from traces](../tutorials/301_calibrate_from_traces.py):
   fit real efficiency factors with confidence intervals
4. [401 — tail-aware sizing](../tutorials/401_tail_aware_sizing.py):
   arrival rates, p99 targets, and headroom you must pay for

Install: `git clone https://github.com/reckon-research/berth && cd berth
&& pip install -e .` — the core is stdlib-only by design.

## The index

berth is built and maintained by [Reckon Research](https://reckonresearch.com),
publisher of the placement-premium index: measured, workload-conditional
premiums across silicon and providers, with confidence intervals and raw
traces attached. The current release is a simulation-validated reference
implementation; measured hardware calibration is published with the index.
Contribute traces via the
[calibration-data template](../.github/ISSUE_TEMPLATE/calibration_data.md) —
submissions pass the same physics gates as our own.

Questions, disputes, methodology arguments: open an issue. Estimate
disputes with traces attached get answered first — see
[CONTRIBUTING](../CONTRIBUTING.md).
