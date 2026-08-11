"""Tests for the savings ledger.

Every test here is about refusing to overstate. A tool that reports what it
has been worth is where this kind of system usually starts lying, and the
lies are predictable: counting money nobody took, counting the same dollar in
two periods, and calling an estimate a measurement.
"""

from datetime import UTC, datetime, timedelta

import pytest

from berth.agent import AgentState, Decision, Outcome, ProposalRecord
from berth.ledger import ClassEconomics, build_ledger, daily_report

NOW = datetime(2026, 8, 11, 9, 0, tzinfo=UTC)


def _iso(d):
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def _fixture(outcome, days_ago=30, note=""):
    st = AgentState(records=[ProposalRecord(
        "voice", "h100-pcie", "mi300x", "corpus_cell",
        _iso(NOW - timedelta(days=days_ago)), outcome,
        _iso(NOW - timedelta(days=days_ago - 1)) if outcome != Outcome.OPEN else "",
        note)])
    econ = [ClassEconomics("voice", 8.0, 4.12)]
    dec = {"voice": Decision("mi300x", "h100-pcie", 2.5, 4.12, 0.106, 2, 0, True)}
    return build_ledger(st, econ, decisions=dec, as_of=_iso(NOW))


def test_an_open_proposal_has_saved_nothing():
    """The single most tempting lie this system could tell. A proposal nobody
    merged is money on the table, and the workload is still on the old
    placement."""
    led = _fixture(Outcome.OPEN)
    assert led.realized() == 0.0
    assert led.available() > 0
    assert "Nothing has been saved here" in daily_report(
        ledger=led, classes=[1], sources_polled=5)


def test_a_declined_proposal_is_not_counted_as_saved_or_lost():
    """The customer had a reason, usually one that has nothing to do with
    cost. It is reported because a constraint has a price."""
    led = _fixture(Outcome.REJECTED, note="AMD quota not approved")
    assert led.realized() == 0.0
    assert led.foregone() > 0


def test_a_class_left_alone_contributes_nothing():
    """The gap between a correct placement and the worst in the fleet is not a
    saving, it is a comparison nobody was going to make. Reporting it would be
    reporting our own existence as value."""
    led = build_ledger(AgentState(), [ClassEconomics("voice", 8.0, 4.12)],
                       as_of=_iso(NOW))
    assert led.realized() == 0.0 and led.available() == 0.0
    text = daily_report(ledger=led, classes=[1], sources_polled=5)
    assert "reporting our own existence as value" in text


def test_the_same_dollar_is_not_counted_in_two_periods():
    """A change merged in March did not earn its March money again in August.
    Windowed figures accrue inside the window only."""
    led = _fixture(Outcome.MERGED, days_ago=95)
    day, week, month, quarter, total = (led.realized("day"),
                                        led.realized("week"),
                                        led.realized("month"),
                                        led.realized("quarter"),
                                        led.realized())
    assert day < week < month < quarter <= total
    assert week == pytest.approx(day * 7, rel=0.01)


def test_realized_accrues_from_the_merge_not_from_the_proposal():
    """The saving starts when the workload moves, not when we suggested it."""
    early = _fixture(Outcome.MERGED, days_ago=60)
    late = _fixture(Outcome.MERGED, days_ago=10)
    assert early.realized() > late.realized()


def test_nothing_is_verified_until_a_receipt_settles_it():
    """An estimate presented as a measurement is the one unrecoverable error
    in this project, and this is where it would happen."""
    led = _fixture(Outcome.MERGED)
    assert led.verified_share == 0.0
    assert all(e.provenance == "ESTIMATED" for e in led.entries)
    text = daily_report(ledger=led, classes=[1], sources_polled=5)
    assert "0% of that is VERIFIED" in text


def test_a_settled_receipt_marks_the_entry_verified():
    st = AgentState(records=[ProposalRecord(
        "voice", "h100-pcie", "mi300x", "corpus_cell",
        _iso(NOW - timedelta(days=40)), Outcome.MERGED,
        _iso(NOW - timedelta(days=39)))])
    dec = {"voice": Decision("mi300x", "h100-pcie", 2.5, 4.12, 0.106, 2, 0, True)}
    receipts = [{"declaration": {"workload_class": "voice"}, "billable": True,
                 "trace_pointer": "s3://traces/2026-07/voice"}]
    led = build_ledger(st, [ClassEconomics("voice", 8.0, 4.12)],
                       decisions=dec, receipts=receipts, as_of=_iso(NOW))
    assert led.verified_share == 1.0
    assert led.entries[0].receipt.startswith("s3://")


def test_a_class_with_no_declared_volume_is_skipped_not_guessed():
    """A percentage is not money until somebody says how much work there is,
    and inferring it would let the reported saving be adjusted by changing an
    assumption."""
    st = AgentState(records=[ProposalRecord(
        "unknown", "a", "b", "t", _iso(NOW), Outcome.MERGED, _iso(NOW))])
    led = build_ledger(st, [ClassEconomics("voice", 8.0, 4.12)],
                       as_of=_iso(NOW))
    assert led.entries == []


def test_a_pass_that_reached_nothing_says_so_before_any_number():
    """An absence of triggers means nothing when no source answered, and the
    report must not let that read as a quiet day."""
    led = _fixture(Outcome.MERGED)
    text = daily_report(ledger=led, classes=[1], sources_polled=0,
                        unreachable={"model:x": "HTTP 403"})
    assert text.index("DID NOT RUN CLEANLY") < text.index("SAVINGS")


def test_a_quiet_day_still_says_what_was_checked():
    """A team that hears nothing for three weeks does not conclude all is
    well, they conclude it stopped running. Silence is correct for action and
    wrong for communication."""
    led = build_ledger(AgentState(), [ClassEconomics("voice", 8.0, 4.12)],
                       as_of=_iso(NOW))
    text = daily_report(ledger=led, classes=[1, 2, 3], sources_polled=5,
                        triggers=7, proposals=0)
    assert "Checked 3 workload classes against 5 sources" in text
    assert "holding its service level" in text
    assert "7 triggers evaluated" in text
