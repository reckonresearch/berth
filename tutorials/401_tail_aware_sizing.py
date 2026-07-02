"""Tutorial 401: arrival rate + p99 SLO -> fleet sizing with headroom priced in.

Run: python tutorials/401_tail_aware_sizing.py
Expected: replicas/utilization/p99 per silicon; cost includes queueing headroom.
"""
from berth import MODELS, PlacementClient, SimBackend, WorkloadSpec

client = PlacementClient(SimBackend(seed=0))
sig = client.profile(WorkloadSpec(model=MODELS["llama3-70b"], target_batch=16,
                                  avg_prompt_tokens=1024, avg_output_tokens=256,
                                  p99_ttft_ms=500.0, arrival_rps=40.0))
for e in client.estimate(sig):
    if e.feasible:
        print(f"{e.silicon:<10} {e.replicas:>3} replicas  util {e.utilization:4.0%}  "
              f"p99 TTFT {e.p99_ttft_ms:4.0f}ms  ${e.cost_per_mtok:.2f}/Mtok")
