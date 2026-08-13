"""Tests for the execution layer.

Every test here is about a guardrail refusing. An execution path is the one
place in this system where a wrong answer moves production, so the tests are
weighted toward what must not happen rather than what should.
"""

from datetime import UTC, datetime, timedelta

from berth.agent import Decision
from berth.execute import (
    Autonomy,
    AutonomyPolicy,
    ExecutionRecord,
    ExecutionState,
    may_execute,
    should_roll_back,
)

NOW = datetime(2026, 8, 12, 14, 0, tzinfo=UTC)   # a Wednesday


def _d(margin=0.31, band=0.106, measured=2, switching=0.03):
    """A decision with a given margin over a $1.00 incumbent."""
    return Decision(recommended="mi300x", incumbent="h100-pcie",
                    recommended_cost=1.0 - margin, incumbent_cost=1.0,
                    confidence_band=band, measured_cells=measured,
                    prior_cells=0, feasible=True, switching_cost=switching)


def _p(**kw):
    base = dict(level=Autonomy.EXECUTE)
    base.update(kw)
    return AutonomyPolicy(**base)


# ------------------------------------------------------------ autonomy level

def test_propose_is_the_default_and_executes_nothing():
    """A policy nobody thought about must behave like one that says ask."""
    ok, why = may_execute(_d(), AutonomyPolicy(), ExecutionState(), "v", now=NOW)
    assert not ok and "pull request" in why
    assert AutonomyPolicy().level == Autonomy.PROPOSE


def test_frozen_does_nothing_at_all():
    ok, why = may_execute(_d(), _p(level=Autonomy.FROZEN), ExecutionState(),
                          "v", now=NOW)
    assert not ok and "frozen" in why


def test_execute_permits_a_clean_decision():
    ok, why = may_execute(_d(), _p(), ExecutionState(), "v", now=NOW)
    assert ok, why


# ------------------------------------------------------------------ the bars

def test_the_bar_to_act_is_higher_than_the_bar_to_ask():
    """A margin can clear the hurdle for proposing and not the hurdle for
    moving production without being asked. That gap is deliberate."""
    d = _d(margin=0.18)
    assert d.clears_band, "this would be proposed"
    ok, why = may_execute(d, _p(min_margin_to_execute=0.25), ExecutionState(),
                          "v", now=NOW)
    assert not ok
    assert "bar for proposing" in why


def test_an_unmeasured_placement_is_never_executed_by_default():
    """A spec-sheet prior has been observed above 40 percent error. Moving
    production onto one unattended is the worst thing this system could do."""
    d = _d(measured=0)
    assert not d.rests_on_measurement
    ok, why = may_execute(d, _p(), ExecutionState(), "v", now=NOW)
    assert not ok and "never been measured" in why

    ok2, _ = may_execute(d, _p(allow_unmeasured=True), ExecutionState(),
                         "v", now=NOW)
    assert ok2, "an explicit opt-in is still available"


def test_a_margin_inside_the_hurdle_is_not_executable():
    d = _d(margin=0.05, band=0.106)
    assert not d.clears_band
    ok, why = may_execute(d, _p(min_margin_to_execute=0.0), ExecutionState(),
                          "v", now=NOW)
    assert not ok and "confidence band" in why


# ------------------------------------------------------------- rate and scope

def test_a_class_cannot_move_twice_inside_the_rate_limit():
    """A placement that moves twice in a week is oscillating, not adapting."""
    st = ExecutionState(records=[ExecutionRecord(
        "v", "a", "b", (NOW - timedelta(hours=20)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        True, "earlier move")])
    ok, why = may_execute(_d(), _p(min_hours_between_moves=168), st, "v", now=NOW)
    assert not ok and "oscillating" in why


def test_the_rate_limit_expires():
    st = ExecutionState(records=[ExecutionRecord(
        "v", "a", "b", (NOW - timedelta(hours=200)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        True, "old move")])
    ok, _ = may_execute(_d(), _p(min_hours_between_moves=168), st, "v", now=NOW)
    assert ok


def test_blast_radius_caps_a_single_pass():
    """One bad corpus update must not be able to move a fleet."""
    ok, why = may_execute(_d(), _p(max_moves_per_pass=1), ExecutionState(),
                          "v", now=NOW, moves_already=1)
    assert not ok and "move a fleet" in why


def test_a_refused_class_does_not_consume_the_blast_radius():
    """Only executions count against it, or one refusal would block the rest."""
    st = ExecutionState(records=[ExecutionRecord(
        "other", "a", "b", NOW.strftime("%Y-%m-%dT%H:%M:%SZ"), False, "refused")])
    assert st.moves_this_pass(NOW - timedelta(hours=1)) == 0


# ------------------------------------------------------------ time and freeze

def test_a_blackout_date_stops_everything():
    """A freeze is a real thing in most organisations, and a placement system
    that ignores one gets uninstalled."""
    ok, why = may_execute(_d(), _p(blackout_dates=("2026-08-12",)),
                          ExecutionState(), "v", now=NOW)
    assert not ok and "blackout" in why


def test_window_mode_refuses_outside_its_windows():
    # Wednesday is weekday 2. This window is Sunday only.
    ok, why = may_execute(_d(), _p(level=Autonomy.WINDOW, windows=((6, 2, 5),)),
                          ExecutionState(), "v", now=NOW)
    assert not ok and "outside the declared execution windows" in why


def test_window_mode_permits_inside_one():
    ok, _ = may_execute(_d(), _p(level=Autonomy.WINDOW, windows=((2, 13, 16),)),
                        ExecutionState(), "v", now=NOW)
    assert ok


def test_window_mode_with_no_windows_permits_nothing():
    """Configuration error rather than a refusal, and it is named as one."""
    ok, why = may_execute(_d(), _p(level=Autonomy.WINDOW), ExecutionState(),
                          "v", now=NOW)
    assert not ok and "configuration error" in why


# ---------------------------------------------------------------- rollback

def test_a_breached_service_level_reverts_regardless_of_savings():
    """A placement that misses its bound is not a cheap placement. It is not
    a placement."""
    back, why = should_roll_back(_p(), breaches=1, observed_cost=0.10,
                                 expected_cost=1.00)
    assert back and "misses its bound" in why


def test_a_cost_estimate_that_did_not_survive_contact_reverts():
    back, why = should_roll_back(_p(), breaches=0, observed_cost=1.40,
                                 expected_cost=1.00)
    assert back and "model was wrong" in why


def test_a_move_that_worked_does_not_revert():
    back, _ = should_roll_back(_p(), breaches=0, observed_cost=0.72,
                               expected_cost=0.70)
    assert not back


def test_rollback_tolerance_is_not_zero():
    """An estimate a few percent optimistic is an estimate, not a failure.
    Reverting on noise would make the system oscillate through its own
    rollback path."""
    back, _ = should_roll_back(_p(), breaches=0, observed_cost=1.10,
                               expected_cost=1.00, tolerance=0.15)
    assert not back


# ------------------------------------------------------------------- the gate

def test_no_guardrail_can_grant_permission():
    """Every check can only say no. There is no path where a guardrail
    permits something the autonomy level did not already allow."""
    for lvl in (Autonomy.FROZEN, Autonomy.PROPOSE):
        ok, _ = may_execute(_d(margin=0.99), _p(level=lvl,
                                                min_margin_to_execute=0.0,
                                                allow_unmeasured=True),
                            ExecutionState(), "v", now=NOW)
        assert not ok, f"{lvl} must not be overridable by permissive guardrails"
