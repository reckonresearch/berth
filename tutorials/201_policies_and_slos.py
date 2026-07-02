"""Tutorial 201: placement policies — objectives and constraints are plain Python.

Run: python tutorials/201_policies_and_slos.py
Expected: two placements; the SLO-constrained one may pick faster, pricier silicon.
"""
from berth import MODELS, PlacementClient, PlacementPolicy, SimBackend, WorkloadSpec, min_cost

client = PlacementClient(SimBackend(seed=0))
sig = client.profile(WorkloadSpec(model=MODELS["llama3-70b"], target_batch=16,
                                  avg_prompt_tokens=1024, avg_output_tokens=256))

cheapest = client.place(sig, PlacementPolicy(objective=min_cost))
print(f"min-cost:        {cheapest.silicon} @ ${cheapest.estimate.cost_per_mtok:.2f}/Mtok")

snappy = client.place(sig, PlacementPolicy(
    objective=min_cost,
    constraints=(lambda e: e.ttft_ms < 300,),   # any Callable[[Estimate], bool]
))
print(f"TTFT<300ms:      {snappy.silicon} @ ${snappy.estimate.cost_per_mtok:.2f}/Mtok "
      f"(TTFT {snappy.estimate.ttft_ms:.0f}ms)")
