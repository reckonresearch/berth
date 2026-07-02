"""Demo: the user-side policy loop.

Everything below the imports is USER code — the platform never sees the
objective except as a callable. Scenario: serve Llama-3-70B chat traffic,
minimize $/Mtok subject to a TTFT SLO, then survive a spot-price spike.
"""

from berth import MODELS, PlacementClient, PlacementPolicy, SimBackend, WorkloadSpec, min_cost

backend = SimBackend(seed=42)
client = PlacementClient(backend)

# 1. profile ----------------------------------------------------------------
workload = WorkloadSpec(
    model=MODELS["llama3-70b"],
    avg_prompt_tokens=1024,
    avg_output_tokens=256,
    target_batch=16,
    p99_ttft_ms=500.0,
)
sig = client.profile(workload)
print(f"workload: {sig.model.name}  batch={sig.batch}  decode AI={sig.decode_ai:.0f} FLOPs/byte\n")

# 2. estimate — the placement premium table ---------------------------------------
print(f"{'silicon':<10} {'ok':<3} {'dev':>3} {'TTFT ms':>8} {'TPOT ms':>8} {'tok/s':>7} "
      f"{'$/Mtok':>8} {'tax':>7}  bound/reason")
for e in client.estimate(sig):
    if e.feasible:
        print(f"{e.silicon:<10} {'y':<3} {e.n_devices:>3} {e.ttft_ms:>8.0f} {e.tpot_ms:>8.1f} "
              f"{e.tokens_per_s:>7.0f} {e.cost_per_mtok:>8.2f} {e.placement_premium:>6.0%}  {e.bound}")
    else:
        print(f"{e.silicon:<10} {'n':<3} {'-':>3} {'-':>8} {'-':>8} {'-':>7} {'-':>8} {'-':>7}  {e.reason}")

# 3. place — user-defined policy ---------------------------------------------
policy = PlacementPolicy(
    objective=min_cost,
    constraints=(lambda e: e.ttft_ms < 500,),
    min_improvement=0.15,
)
h = client.place(sig, policy)
print(f"\nplaced on {h.silicon} x{h.estimate.n_devices} @ ${h.estimate.cost_per_mtok:.2f}/Mtok")

# 4. migrate — market moves against us ---------------------------------------
backend.shock(h.silicon, 2.2)
print(f"[market] spot spike: {h.silicon} price x2.2")
h2 = client.migrate(h)
if h2.silicon != h.silicon:
    print(f"migrated -> {h2.silicon} x{h2.estimate.n_devices} @ ${h2.estimate.cost_per_mtok:.2f}/Mtok")
else:
    print(f"held {h2.silicon} (inside hysteresis) @ ${h2.estimate.cost_per_mtok:.2f}/Mtok")

# Small drift doesn't cause churn.
backend.tick(3)
h3 = client.migrate(h2)
print(f"after 3 market ticks: {'held ' + h3.silicon if h3.silicon == h2.silicon else 'moved -> ' + h3.silicon}")
