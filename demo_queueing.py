"""Demo: does the tail change the ranking?

Same Llama-70B workload, now with real traffic (arrival rate) and a p99 TTFT
SLO. Compares mean-based ranking (v0.2 semantics) against tail-aware ranking
(fleet sized for the SLO, cost includes queueing headroom).
"""

from berth import MODELS, PlacementClient, SimBackend, WorkloadSpec

client = PlacementClient(SimBackend(seed=42))

base = dict(model=MODELS["llama3-70b"], avg_prompt_tokens=1024,
            avg_output_tokens=256, target_batch=16)

# --- v0.2 semantics: mean latencies, no traffic model -----------------------
mean_sig = client.profile(WorkloadSpec(**base, p99_ttft_ms=500.0))
print("mean-based ranking (single replica, SLO checked against MEANS):")
for e in client.estimate(mean_sig):
    if e.feasible:
        print(f"  {e.silicon:<10} ${e.cost_per_mtok:>5.2f}/Mtok   tax {e.placement_premium:>4.0%}")

# --- tail-aware: 40 rps, p99 TTFT <= 500ms ----------------------------------
tail_sig = client.profile(WorkloadSpec(**base, p99_ttft_ms=500.0, arrival_rps=40.0))
print("\ntail-aware ranking (40 rps, p99 TTFT <= 500ms, headroom priced in):")
print(f"  {'silicon':<10} {'repl':>4} {'util':>6} {'p99TTFT':>8} {'$/Mtok':>8} {'tax':>6}")
for e in client.estimate(tail_sig):
    if e.feasible:
        print(f"  {e.silicon:<10} {e.replicas:>4} {e.utilization:>6.0%} "
              f"{e.p99_ttft_ms:>7.0f}m {e.cost_per_mtok:>8.2f} {e.placement_premium:>5.0%}")
    else:
        print(f"  {e.silicon:<10}  infeasible: {e.reason}")

# --- the flip: tighten the SLO to 300ms -------------------------------------
tight_sig = client.profile(WorkloadSpec(**base, p99_ttft_ms=300.0, arrival_rps=40.0))
print("\ntight SLO (p99 TTFT <= 300ms) — cheapest silicon changes:")
for e in client.estimate(tight_sig):
    if e.feasible:
        print(f"  {e.silicon:<10} {e.replicas:>4} repl  p99 {e.p99_ttft_ms:>4.0f}ms  "
              f"${e.cost_per_mtok:.2f}/Mtok  tax {e.placement_premium:.0%}")
    else:
        print(f"  {e.silicon:<10} infeasible: {e.reason}")
