"""Demo: per-class calibration + drift detection.

Hidden reality: (1) MI300X long-context attention runs at bw_eff 0.42 vs 0.60
short-context — invisible to a global fit; (2) over the observation span,
MI300X bw_eff trends 0.50 -> 0.62 (software stack maturing). The calibrator
and drift detector see only priors + noisy traces.
"""

from berth import FLEET, MODELS, PlacementClient, SimBackend, WorkloadSpec
from berth.classes import calibrate_classed, detect_drift, workload_class
from berth.traces import generate_traces


# --- 1. class structure ------------------------------------------------------
def class_eff(name, sig, t):
    hw = FLEET[name]
    if name != "mi300x":
        return hw.mfu, hw.bw_eff
    return hw.mfu, (0.42 if "ctxL" in workload_class(sig) else 0.60)

traces = generate_traces(FLEET, n_per_silicon=150, noise_sigma=0.05, seed=11,
                         eff_fn=class_eff)
cf = calibrate_classed(FLEET, traces)

print("mi300x per-class bw_eff fits (global fit would blur these together):")
cells = sorted((k[1], v[1], cf.class_counts[k]) for k, v in cf.class_fits.items()
               if k[0] == "mi300x")
for cls, bw, n in cells:
    marker = " <- degraded long-context path" if "ctxL" in cls else ""
    print(f"  {cls:<12} bw_eff {bw:.3f}  (n={n}){marker}")
g = cf.global_fits["mi300x"]
print(f"  {'GLOBAL':<12} bw_eff {g.bw_eff:.3f}  — the average that hides the split")

# Decision impact: same silicon, different workload class, different truth.
client = PlacementClient(SimBackend(seed=1), profile_resolver=cf.resolver())
for label, prompt in (("short-ctx chat", 512), ("long-ctx RAG", 8192)):
    sig = client.profile(WorkloadSpec(model=MODELS["llama3-70b"], target_batch=16,
                                      avg_prompt_tokens=prompt, avg_output_tokens=256))
    e = next(x for x in client.estimate(sig) if x.silicon == "mi300x" and x.feasible)
    print(f"  {label:<16} mi300x ${e.cost_per_mtok:.2f}/Mtok  premium {e.placement_premium:.0%}")

# --- 2. drift ----------------------------------------------------------------
def maturing(name, sig, t):
    hw = FLEET[name]
    return (hw.mfu, 0.50 + 0.12 * t) if name == "mi300x" else (hw.mfu, hw.bw_eff)

drift_traces = generate_traces(FLEET, n_per_silicon=120, noise_sigma=0.05,
                               seed=13, eff_fn=maturing)
print("\ndrift signals (windowed refits, OLS slope over observation span):")
for s in detect_drift(FLEET, drift_traces, n_windows=5, threshold=0.05):
    if s.flagged:
        print(f"  FLAG {s.silicon} {s.param}: {s.start:.3f} -> {s.end:.3f} "
              f"(slope {s.slope:+.3f}/span) — placement premium is moving")
flagged = {(s.silicon, s.param) for s in
           detect_drift(FLEET, drift_traces, n_windows=5) if s.flagged}
print(f"  stable silicon flagged: {len([f for f in flagged if f[0] != 'mi300x'])} (expect 0)")
