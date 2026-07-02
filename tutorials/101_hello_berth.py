"""Tutorial 101: profile a workload, estimate it across the fleet.

Run: python tutorials/101_hello_berth.py
Expected: a premium-annotated table; cheapest feasible silicon has premium 0%.
"""
from berth import MODELS, PlacementClient, SimBackend, WorkloadSpec

client = PlacementClient(SimBackend(seed=0))
sig = client.profile(WorkloadSpec(model=MODELS["llama3-8b"], target_batch=8))
for e in client.estimate(sig):
    if e.feasible:
        print(f"{e.silicon:<10} ${e.cost_per_mtok:6.2f}/Mtok  "
              f"premium {e.placement_premium:5.0%}  {e.bound}-bound")
