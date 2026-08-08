"""Tests for the holdout protocol implementation.

The protocol's integrity rests on properties rather than on behaviour: the
assignment must be reproducible, the sample must not be choosable, the holdout
cost must appear, and a loss must carry. Each of those is a test here.
"""

import uuid

import pytest

from berth.holdout import (
    Declaration,
    LegObservations,
    ProtocolError,
    assign,
    breaker_tripped,
    decile_divergence,
    realised_fraction,
    scale_floor,
    stationarity,
    warmup_complete,
    wilson_interval,
)
from berth.receipt import build_receipt, settle


def _declaration(**kw):
    base = dict(workload_class="voice", baseline_placement="4x H100 PCIe",
                treatment_placement="6x L40S", slo_metric="p99_ttft_ms",
                slo_bound_ms=800, quality_floor="valid JSON",
                holdout_fraction=0.05, seed="seed-committed-in-advance",
                annual_class_spend=480_000)
    base.update(kw)
    return Declaration(**base)


def _leg(spend, tokens, requests=1_000_000, compliant=950_000, **kw):
    d = dict(spend=spend, requests=requests, compliant_requests=compliant,
             served_tokens=tokens, first_half_requests=requests // 2,
             second_half_requests=requests // 2,
             first_half_cost_per_mtok=1.0, second_half_cost_per_mtok=1.0,
             median_context=1500, median_generation=120,
             context_deciles=[300, 600, 900, 1200, 1500, 1800, 2400, 3200,
                              4800, 7600])
    d.update(kw)
    return LegObservations(**d)


# --------------------------------------------------------------- assignment

def test_assignment_is_deterministic_from_the_published_seed():
    """Either party can recompute any request's leg. That is what makes a
    dispute resolvable by rerunning arithmetic rather than by argument."""
    rid = "req-abc-123"
    assert assign(rid, "s", 0.05) == assign(rid, "s", 0.05)


def test_a_different_seed_gives_a_different_partition():
    """If the seed did not matter, committing it in advance would not be a
    defence against anything."""
    ids = [str(uuid.uuid4()) for _ in range(2000)]
    a = {r for r in ids if assign(r, "seed-one", 0.5) == "baseline"}
    b = {r for r in ids if assign(r, "seed-two", 0.5) == "baseline"}
    overlap = len(a & b) / len(a)
    assert 0.3 < overlap < 0.7, overlap


def test_assignment_hits_the_declared_fraction():
    ids = [str(uuid.uuid4()) for _ in range(50_000)]
    assert abs(realised_fraction(ids, "s", 0.05) - 0.05) < 0.005


def test_structured_identifiers_still_distribute():
    """Sequential and tenant-prefixed identifiers carry structure in their
    leading bytes. The seed is prepended so that structure cannot survive into
    the assignment, which would put one tenant disproportionately on one leg
    and make the measured difference a property of the traffic."""
    seq = [f"req-{i:09d}" for i in range(50_000)]
    tenant = [f"acme-corp-{i:06d}" for i in range(25_000)] + \
             [f"globex-inc-{i:06d}" for i in range(25_000)]
    for ids, label in ((seq, "sequential"), (tenant, "tenant-prefixed")):
        got = realised_fraction(ids, "s", 0.05)
        assert abs(got - 0.05) < 0.005, f"{label}: {got}"
    # And each tenant individually, which is the case that matters.
    acme = realised_fraction([i for i in tenant if i.startswith("acme")],
                             "s", 0.05)
    assert abs(acme - 0.05) < 0.01, acme


def test_an_empty_seed_is_refused():
    with pytest.raises(ProtocolError, match="selective measurement"):
        assign("r", "", 0.05)


def test_degenerate_fractions_are_refused():
    for f in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ProtocolError):
            assign("r", "s", f)


# -------------------------------------------------------------- instrument

def test_scale_floor_matches_the_published_figure():
    assert scale_floor() == pytest.approx(333_333, rel=1e-3)


def test_instrument_follows_from_spend_not_negotiation():
    assert _declaration(annual_class_spend=100_000).instrument == "assurance_light"
    assert _declaration(annual_class_spend=300_000).instrument == "assurance"
    assert _declaration(annual_class_spend=480_000).instrument == "savings_share"


# -------------------------------------------------------------- settlement

def test_holdout_cost_is_deducted_not_netted_silently():
    """The holdout runs on the worse placement by design and costs the
    customer real money. Hiding it would make every receipt an
    overstatement."""
    d = _declaration()
    s = settle(d, _leg(2_000, 485_000_000), _leg(38_000, 14_232_000_000))
    assert s.holdout_cost > 0
    assert s.net_saving == pytest.approx(s.gross_saving - s.holdout_cost)


def test_a_negative_period_produces_no_invoice():
    d = _declaration()
    s = settle(d, _leg(2_000, 500_000_000), _leg(38_000, 8_000_000_000))
    assert s.net_saving < 0
    assert s.invoice == 0.0


def test_a_loss_carries_forward():
    """Capturing a share of every gain and none of any loss is an asymmetry a
    counterparty is right to refuse. Carrying it forward costs nothing when
    the recommendation was correct."""
    d = _declaration()
    base, treat = _leg(2_000, 485_000_000), _leg(38_000, 14_232_000_000)
    clean = settle(d, base, treat)
    carried = settle(d, base, treat, carry_forward=10_000)
    assert carried.billable_saving == pytest.approx(clean.net_saving - 10_000)
    assert carried.invoice < clean.invoice


def test_billing_uses_the_conservative_end_of_the_interval():
    d = _declaration()
    s = settle(d, _leg(2_000, 485_000_000), _leg(38_000, 14_232_000_000))
    assert s.invoice_conservative <= s.invoice


# ------------------------------------------------------------ stationarity

def test_a_load_shift_blocks_billing():
    """Traffic that grew during the window is not a placement result."""
    d = _declaration()
    base = _leg(2_000, 485_000_000, first_half_requests=300_000,
                second_half_requests=700_000)
    r = build_receipt(d, base, _leg(38_000, 14_232_000_000),
                      trace_pointer="x", serving_stack={})
    assert not r.billable
    assert "stationarity" in r.reason_not_billable


def test_a_mix_shift_flags_without_voiding():
    """A median can hold while the distribution moves underneath it. That
    explains a difference without necessarily invalidating one, so it is a
    decision rather than a void."""
    base = _leg(2_000, 485_000_000)
    treat = _leg(38_000, 14_232_000_000,
                 context_deciles=[300, 600, 900, 1200, 1500, 2400, 4800,
                                  7000, 9000, 12000])
    checks, blocking = stationarity(base, treat)
    assert not checks["4.3b_context_mix"][1], "the mix shift must be seen"
    assert not blocking, "but it must not void the period on its own"


def test_decile_divergence_is_relative_not_a_step_function():
    """Kolmogorov-Smirnov on ten boundary values moves in steps of 0.1
    regardless of how close the distributions are, which puts every
    comparison on a threshold. Comparing boundaries directly is both the
    right object and interpretable."""
    a = [300, 600, 900, 1200, 1500]
    near = [305, 610, 905, 1210, 1520]
    far = [300, 600, 900, 1200, 4000]
    assert decile_divergence(a, near) < 0.05
    assert decile_divergence(a, far) > 0.5


# ------------------------------------------------------- warm-up, breaker

def test_warmup_requires_stability_not_a_fixed_count():
    """A cold placement has empty caches and uncaptured graphs. Comparing a
    warm baseline against a cold treatment measures the start."""
    assert not warmup_complete([10.0, 6.0, 4.2])
    assert warmup_complete([4.2, 4.3])
    assert not warmup_complete([4.2])


def test_breaker_trips_only_above_the_margin_and_the_request_floor():
    d = _declaration()                       # bound 800 ms, margin 20 percent
    assert not breaker_tripped(d, 900, 5_000)       # within margin
    assert not breaker_tripped(d, 2_000, 50)        # too few requests to judge
    assert breaker_tripped(d, 1_000, 5_000)         # breach, enough evidence


def test_a_tripped_breaker_voids_rather_than_counting_as_a_loss():
    """A placement that never held the bound was never a measurement, so it
    must not carry forward against the next period."""
    d = _declaration()
    r = build_receipt(d, _leg(2_000, 485_000_000), _leg(38_000, 8_000_000_000),
                      trace_pointer="x", serving_stack={},
                      breaker_tripped=True)
    assert not r.billable
    assert "void" in r.reason_not_billable


# ---------------------------------------------------------------- receipt

def test_receipt_carries_provenance_on_every_class_of_figure():
    d = _declaration()
    r = build_receipt(d, _leg(2_000, 485_000_000), _leg(38_000, 14_232_000_000),
                      trace_pointer="s3://traces/x", serving_stack={"b": "vLLM"})
    for key in ("spend", "served_tokens", "compliance", "interval", "saving",
                "assignment"):
        assert key in r.provenance
    assert r.billable
    assert "s3://" in r.to_json()


def test_wilson_interval_stays_inside_the_unit_interval_at_the_edges():
    """Compliance legitimately sits near zero and near one, where a normal
    approximation produces bounds outside [0, 1] and stops meaning
    anything."""
    for lo, hi in (wilson_interval(0, 100), wilson_interval(100, 100),
                   wilson_interval(1, 3)):
        assert 0.0 <= lo <= hi <= 1.0
