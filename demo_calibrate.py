"""Demo: calibration loop.

A hidden 'real world' fleet generates noisy traces; the calibrator sees only
priors + traces. Shows parameter recovery, holdout MAPE, and — the part that
matters commercially — whether better parameters change the placement decision.
"""

from berth import (
    FLEET,
    MODELS,
    PlacementClient,
    PlacementPolicy,
    SimBackend,
    WorkloadSpec,
    min_cost,
)
from berth.calibrate import calibrate
from berth.traces import generate_traces, make_true_fleet

# Hidden truth (unknown to the fitter). MI300X software gap worse than prior;
# H100 bandwidth efficiency better than prior.
# This is a SUBSET of FLEET on purpose: the demo can only report recovery for
# silicon whose truth it invented. Reporting loops below iterate TRUE_EFF, not
# FLEET, so adding an accelerator to the registry never breaks this demo.
TRUE_EFF = {
    "h100-sxm": (0.44, 0.82), "h200-sxm": (0.55, 0.70), "mi300x": (0.30, 0.50),
    "a100-80g": (0.52, 0.78), "l40s": (0.45, 0.70), "cpu-spr": (0.25, 0.55),
}

true_fleet = make_true_fleet(FLEET, TRUE_EFF)
traces = generate_traces(true_fleet, n_per_silicon=60, noise_sigma=0.05, seed=7)
print(f"generated {len(traces)} traces (5% measurement noise)\n")

calibrated, report = calibrate(FLEET, traces)

print(f"{'silicon':<10} {'n':>4}   {'mfu prior->fit (true)':<26} {'bw_eff prior->fit (true)':<26}")
for name in TRUE_EFF:
    p = FLEET[name]
    if name not in report.fitted:
        continue
    f_mfu, f_bw = report.fitted[name]
    t_mfu, t_bw = TRUE_EFF[name]
    print(f"{name:<10} {report.n_traces[name]:>4}   "
          f"{p.mfu:.2f} -> {f_mfu:.3f} ({t_mfu:.2f})        "
          f"{p.bw_eff:.2f} -> {f_bw:.3f} ({t_bw:.2f})")

print(f"\nholdout MAPE: prior {report.mape_prior:.1%} -> calibrated {report.mape_calibrated:.1%}")

# Decision impact: same workload, same policy, prior vs calibrated fleet.
sig_args = dict(model=MODELS["llama3-70b"], avg_prompt_tokens=1024,
                avg_output_tokens=256, target_batch=16, p99_ttft_ms=500.0)
policy = PlacementPolicy(objective=min_cost, constraints=(lambda e: e.ttft_ms < 500,))

for label, fleet in (("prior", FLEET), ("calibrated", calibrated)):
    client = PlacementClient(SimBackend(seed=42, fleet=fleet))
    h = client.place(client.profile(WorkloadSpec(**sig_args)), policy)
    print(f"{label:>10} fleet places: {h.silicon} x{h.estimate.n_devices} "
          f"@ ${h.estimate.cost_per_mtok:.2f}/Mtok")
