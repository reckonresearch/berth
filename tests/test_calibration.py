"""Calibration tests: blind recovery of hidden efficiencies, holdout MAPE, integration."""

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
from berth.calibrate import calibrate
from berth.traces import generate_traces, make_true_fleet
from berth.workload import profile

# Hidden ground truth, deliberately off the priors in both directions.
TRUE_EFF = {
    "h100-sxm": (0.44, 0.82),   # prior (0.50, 0.75)
    "h200-sxm": (0.55, 0.70),
    "mi300x":   (0.30, 0.50),   # prior (0.35, 0.65) — software gap worse than assumed
    "a100-80g": (0.52, 0.78),
    "l40s":     (0.45, 0.70),
    "cpu-spr":  (0.25, 0.55),
}


# Pinned rather than taken from FLEET. This fixture asserts a validation
# figure, and a validation figure must not move because unrelated silicon was
# added to the registry. Adding trn1 and trn2 shifted the aggregate MAPE by a
# fraction of a percent and put a 50 percent improvement threshold on the
# wrong side of the line, which looked like a regression and was a fleet
# change. Extend this list deliberately, and expect the numbers to move.
CALIBRATION_FLEET = tuple(TRUE_EFF)


@pytest.fixture(scope="module")


def calibrated():
    subset = {k: FLEET[k] for k in CALIBRATION_FLEET}
    true_fleet = make_true_fleet(subset, TRUE_EFF)
    traces = generate_traces(true_fleet, n_per_silicon=60, noise_sigma=0.05, seed=7)
    fleet, report = calibrate(subset, traces)
    return fleet, report


def test_generator_deterministic():
    tf = make_true_fleet(FLEET, TRUE_EFF)
    a = generate_traces(tf, n_per_silicon=5, seed=3)
    b = generate_traces(tf, n_per_silicon=5, seed=3)
    assert a == b


def test_blind_parameter_recovery(calibrated):
    fleet, report = calibrated
    for name, (true_mfu, true_bw) in TRUE_EFF.items():
        got_mfu, got_bw = report.fitted[name]
        # 5% lognormal noise, median estimator: recovery within 8% relative.
        assert got_mfu == pytest.approx(true_mfu, rel=0.08), name
        assert got_bw == pytest.approx(true_bw, rel=0.08), name


def test_holdout_mape_improves(calibrated):
    _, report = calibrated
    assert report.mape_calibrated < report.mape_prior * 0.5   # at least halved
    assert report.mape_calibrated < 0.10                      # near noise floor


def test_calibrated_fleet_plugs_into_client(calibrated):
    fleet, _ = calibrated
    client = PlacementClient(SimBackend(seed=1, fleet=fleet))
    sig = profile(WorkloadSpec(model=MODELS["llama3-70b"], target_batch=16))
    h = client.place(sig, PlacementPolicy(objective=min_cost))
    assert h.estimate.feasible  # zero policy-code changes required


def test_unobserved_silicon_keeps_prior():
    true_fleet = make_true_fleet(FLEET, TRUE_EFF)
    traces = [t for t in generate_traces(true_fleet, n_per_silicon=30, seed=9)
              if t.silicon != "l40s"]
    fleet, report = calibrate(FLEET, traces)
    assert report.fitted["l40s"] == (FLEET["l40s"].mfu, FLEET["l40s"].bw_eff)
    assert report.n_traces["l40s"] == 0


def test_bootstrap_ci_covers_truth_and_is_tight(calibrated):
    _, report = calibrated
    for name, (true_mfu, true_bw) in TRUE_EFF.items():
        (mfu_lo, mfu_hi), (bw_lo, bw_hi) = report.ci95[name]
        assert mfu_lo <= true_mfu * 1.02 and mfu_hi >= true_mfu * 0.98, name
        assert bw_lo <= true_bw * 1.02 and bw_hi >= true_bw * 0.98, name
        assert (mfu_hi - mfu_lo) / true_mfu < 0.15   # informative, not vacuous
        assert (bw_hi - bw_lo) / true_bw < 0.15
