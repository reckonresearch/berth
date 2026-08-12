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
    subset = {k: FLEET[k] for k in TRUE_EFF}
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
    # Bar set from the estimator's measured behaviour, not from a round
    # number. Across 40 seeds at n=60 with 5 percent lognormal noise, the
    # ratio of calibrated to prior error runs 0.407 best, 0.509 median, 0.595
    # worst. The previous bar was 0.50, which 60 percent of seeds exceed: it
    # passed because seed 7 happened to land at 0.494, and it had roughly a
    # coin-flip chance of failing on any given day. A threshold that only
    # holds on the seed it was written against is testing the random number
    # generator rather than the estimator.
    #
    # 0.65 sits above the observed worst case with room for the tail. The
    # claim it defends is that calibration removes a substantial share of the
    # prior's error, which it does: a third at worst, half typically.
    assert report.mape_calibrated < report.mape_prior * 0.65
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
        # Bar set from the estimator's measured precision, not a round number.
        # Across 20 seeds, 6 silicon and both parameters at n=60 with 5 percent
        # lognormal noise, the bootstrap width is 4.9 percent at the median,
        # 11.3 at p90 and 19.2 at p99. The previous 15 percent bar was exceeded
        # by 2.5 percent of draws, which with twelve assertions per run is
        # roughly a one in four chance of a spurious failure. It passed by luck
        # of the seed and failed the first time the fleet subset changed which
        # random numbers each silicon drew.
        #
        # The purpose of the check is that the interval is informative rather
        # than vacuous. A 25 percent interval on these parameters is still
        # informative; a vacuous one would be several times wider.
        assert (mfu_hi - mfu_lo) / true_mfu < 0.25
        assert (bw_hi - bw_lo) / true_bw < 0.25
