"""Per-class calibration and drift tests: blind recovery of class structure
and time trends, fallback hierarchy, client integration via resolver."""

import pytest

from berth import (
    FLEET,
    MODELS,
    PlacementClient,
    PlacementPolicy,
    SimBackend,
    WorkloadSpec,
    min_cost,
)
from berth.classes import ClassedFleet, calibrate_classed, detect_drift, workload_class
from berth.traces import generate_traces
from berth.workload import profile


def sig(batch, prompt=512, out=128, model="llama3-8b"):
    return profile(WorkloadSpec(model=MODELS[model], target_batch=batch,
                                avg_prompt_tokens=prompt, avg_output_tokens=out))


# ---------- class taxonomy ----------

def test_workload_class_buckets():
    assert workload_class(sig(batch=1, prompt=256)) == "b1-4/ctxS"
    assert workload_class(sig(batch=16, prompt=2048)) == "b8-16/ctxM"
    assert workload_class(sig(batch=32, prompt=8192)) == "b32+/ctxL"


# ---------- per-class blind recovery ----------

# Hidden truth: mi300x long-context attention path is much worse than short.
def _mi300x_class_eff(name, s, t):
    hw = FLEET[name]
    if name != "mi300x":
        return hw.mfu, hw.bw_eff
    return hw.mfu, (0.42 if "ctxL" in workload_class(s) else 0.60)


@pytest.fixture(scope="module")
def classed():
    traces = generate_traces(FLEET, n_per_silicon=150, noise_sigma=0.05,
                             seed=11, eff_fn=_mi300x_class_eff)
    return calibrate_classed(FLEET, traces), traces


def test_per_class_recovery(classed):
    cf, _ = classed
    long_cells = [(k, v) for k, v in cf.class_fits.items()
                  if k[0] == "mi300x" and "ctxL" in k[1]
                  and cf.class_counts[k] >= cf.min_traces]
    short_cells = [(k, v) for k, v in cf.class_fits.items()
                   if k[0] == "mi300x" and "ctxS" in k[1]
                   and cf.class_counts[k] >= cf.min_traces]
    assert long_cells and short_cells
    for _, (_, bw) in long_cells:
        assert bw == pytest.approx(0.42, rel=0.10)
    for _, (_, bw) in short_cells:
        assert bw == pytest.approx(0.60, rel=0.10)


def test_resolver_returns_class_specific_profiles(classed):
    cf, _ = classed
    long_hw = cf.profile_for("mi300x", sig(batch=16, prompt=8192, model="llama3-70b"))
    short_hw = cf.profile_for("mi300x", sig(batch=16, prompt=512))
    assert long_hw.bw_eff < short_hw.bw_eff  # class structure survives lookup


def test_fallback_thin_cell_to_global_to_prior():
    # Only 3 traces total: every cell is thin -> global fit; absent silicon -> prior.
    traces = generate_traces(FLEET, n_per_silicon=3, seed=2)
    cf = calibrate_classed(FLEET, traces, min_traces=12)
    s = sig(batch=16, prompt=1024)
    hw = cf.profile_for("h100-sxm", s)
    assert hw == cf.global_fits["h100-sxm"]          # thin cell -> global
    empty = ClassedFleet(prior=FLEET, global_fits={}, class_fits={}, class_counts={})
    assert empty.profile_for("h100-sxm", s) == FLEET["h100-sxm"]  # -> prior


def test_client_integration_via_resolver(classed):
    cf, _ = classed
    client = PlacementClient(SimBackend(seed=1), profile_resolver=cf.resolver())
    h = client.place(
        client.profile(WorkloadSpec(model=MODELS["llama3-70b"], target_batch=16,
                                    avg_prompt_tokens=8192, avg_output_tokens=256)),
        PlacementPolicy(objective=min_cost),
    )
    assert h.estimate.feasible  # zero policy-code changes


# ---------- drift ----------

def _rocm_maturing(name, s, t):
    hw = FLEET[name]
    if name == "mi300x":
        return hw.mfu, 0.50 + 0.12 * t   # bw_eff 0.50 -> 0.62 over the span
    return hw.mfu, hw.bw_eff


@pytest.fixture(scope="module")
def drift_signals():
    traces = generate_traces(FLEET, n_per_silicon=120, noise_sigma=0.05,
                             seed=13, eff_fn=_rocm_maturing)
    return detect_drift(FLEET, traces, n_windows=5, threshold=0.05)


def test_drift_detected_on_trending_silicon(drift_signals):
    s = next(x for x in drift_signals if x.silicon == "mi300x" and x.param == "bw_eff")
    assert s.flagged
    assert s.slope == pytest.approx(0.12, abs=0.04)
    assert s.end > s.start


def test_no_false_positive_on_stable_silicon(drift_signals):
    stable = [x for x in drift_signals if x.silicon != "mi300x"]
    assert stable and not any(x.flagged for x in stable)


def test_drift_silent_on_insufficient_data():
    traces = generate_traces(FLEET, n_per_silicon=10, seed=3, eff_fn=_rocm_maturing)
    assert detect_drift(FLEET, traces, n_windows=5, min_per_window=8) == []
