# Concepts

## The placement premium
For a given workload, `premium = cost_per_Mtok(here) / cost_per_Mtok(best feasible) - 1`.
It is a property of a (workload, silicon, price, SLO) tuple — not of hardware
alone. The same GPU-hour delivers different useful work per dollar depending
on batch regime, context length, software stack, and the SLO you must hold.
berth estimates it analytically, calibrates it from measured traces, and
annotates every estimate with it.

## The roofline model
Decode step time = max(compute time, memory time). Weights are read once per
step and amortized across the batch; KV reads scale per sequence; prefill
carries a quadratic causal-attention term. Every estimate reports which roof
binds. All formulas are closed-form and derivable by hand — auditability is
a design constraint, not a style choice. Fitted corrections (per-silicon
efficiency factors) layer on top via calibration; they never replace the
analytical core.

## Calibration by inversion
Each measured trace inverts to a parameter estimate (TTFT -> mfu;
memory-bound TPOT -> bw_eff), aggregated with robust medians and reported
with bootstrap 95% CIs. Validation is blind recovery: hidden ground truth,
noisy traces, prove the fitter finds it. See tutorials/301.

## Tail-aware sizing
With an arrival rate and a p99 TTFT SLO, a replica pool is modeled as an
M/M/c queue on batch slots (Erlang-C, closed form). Cost then includes the
headroom you must buy to hold the tail — which can reorder which silicon is
cheapest. See tutorials/401.
