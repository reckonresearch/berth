"""Queueing tests: Erlang C against known values, sizing behavior, compat."""

import math

import pytest

from berth import FLEET, MODELS, PlacementClient, SimBackend, WorkloadSpec
from berth.estimate import estimate
from berth.queueing import erlang_c, p99_wait_s, size_replicas
from berth.workload import profile

# ---------- Erlang C math ----------

def test_erlang_c_known_value_mm1():
    # M/M/1: P(wait) = rho exactly.
    assert erlang_c(1, 0.5) == pytest.approx(0.5)


def test_erlang_c_known_value_textbook():
    # Classic call-center case: c=10, offered load a=8 -> C ~ 0.409.
    assert erlang_c(10, 8.0) == pytest.approx(0.409, abs=0.005)


def test_erlang_c_saturated_returns_one():
    assert erlang_c(4, 4.0) == 1.0
    assert erlang_c(4, 5.0) == 1.0


def test_p99_wait_mm1_closed_form():
    # M/M/1: P(W>t) = rho*exp(-(mu-lam)t) -> p99 = ln(rho/0.01)/(mu-lam).
    mu, lam = 2.0, 1.0
    expected = math.log(0.5 / 0.01) / (mu - lam)
    assert p99_wait_s(1, mu, lam) == pytest.approx(expected)


def test_p99_wait_zero_at_light_load():
    assert p99_wait_s(32, mu=1.0, lam=0.1) == 0.0  # P(wait) < 1%


# ---------- replica sizing ----------

def test_sizing_monotone_in_arrival_rate():
    kw = dict(residence_s=10.0, batch=16, ttft_mean_ms=200, p99_ttft_slo_ms=500)
    r = [size_replicas(lam, **kw).replicas for lam in (2, 8, 32)]
    assert r[0] <= r[1] <= r[2] and r[2] > r[0]


def test_tighter_slo_buys_more_headroom():
    kw = dict(arrival_rps=12.0, residence_s=10.0, batch=16, ttft_mean_ms=200)
    loose = size_replicas(**kw, p99_ttft_slo_ms=2000)
    tight = size_replicas(**kw, p99_ttft_slo_ms=250)
    assert tight.replicas > loose.replicas
    assert tight.utilization < loose.utilization
    assert tight.p99_ttft_ms <= 250


def test_sizing_infeasible_when_slo_below_service_floor():
    # SLO below mean prefill time: no replica count can help.
    s = size_replicas(arrival_rps=5, residence_s=10.0, batch=16,
                      ttft_mean_ms=600, p99_ttft_slo_ms=500)
    assert not s.feasible


# ---------- estimator integration ----------

def _sig(rps, slo=500.0):
    return profile(WorkloadSpec(model=MODELS["llama3-70b"], target_batch=16,
                                avg_prompt_tokens=1024, avg_output_tokens=256,
                                p99_ttft_ms=slo, arrival_rps=rps))


def test_no_arrival_rate_preserves_v02_semantics():
    hw = FLEET["h100-sxm"]
    old = estimate(profile(WorkloadSpec(model=MODELS["llama3-70b"], target_batch=16,
                                        avg_prompt_tokens=1024, avg_output_tokens=256)),
                   hw, hw.base_price_hr)
    assert old.replicas == 1 and old.p99_ttft_ms is None


def test_tail_aware_costs_exceed_mean_based():
    # Headroom is never free: $/Mtok at p99 SLO >= naive single-replica $/Mtok.
    hw = FLEET["h100-sxm"]
    naive = estimate(profile(WorkloadSpec(model=MODELS["llama3-70b"], target_batch=16,
                                          avg_prompt_tokens=1024, avg_output_tokens=256)),
                     hw, hw.base_price_hr)
    tail = estimate(_sig(rps=20.0), hw, hw.base_price_hr)
    assert tail.feasible
    assert tail.cost_per_mtok >= naive.cost_per_mtok
    assert tail.replicas >= 1 and 0 < tail.utilization < 1
    assert tail.p99_ttft_ms <= 500.0


def test_client_end_to_end_with_arrival_rate():
    client = PlacementClient(SimBackend(seed=1))
    ests = [e for e in client.estimate(_sig(rps=25.0)) if e.feasible]
    assert ests
    best = ests[0]
    assert best.placement_premium == pytest.approx(0.0)
    assert all(e.p99_ttft_ms is not None and e.p99_ttft_ms <= 500.0 for e in ests)
