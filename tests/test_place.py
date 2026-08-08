"""Tests for the placement decision record.

A decision is not an estimate. It names what was recommended, what it beat,
by how much, and whether that margin survives the uncertainty on it. A
contract is written against this document, so the properties that matter are
about exclusion and honesty rather than about arithmetic.
"""

import pytest

from berth.place import PRIOR_BAND, decide, render


def _decide(**kw):
    base = dict(workload_class="voice", model_key="llama3-8b",
                incumbent="h100-pcie", slo_bound_ms=1000, batch=8,
                prompt_tokens=512, output_tokens=128)
    base.update(kw)
    return decide(**base)


def test_a_placement_missing_the_bound_is_excluded_not_ranked():
    """Cost per compliant token with no compliant output is not high, it is
    undefined. Ranking it cheaply is what a price sheet does."""
    r = _decide(slo_bound_ms=100)
    ranked = {c["silicon"] for c in r.candidates}
    excluded = {c["silicon"] for c in r.excluded}
    assert "cpu-spr" in excluded
    assert "cpu-spr" not in ranked
    reason = next(c["excluded_reason"] for c in r.excluded
                  if c["silicon"] == "cpu-spr")
    assert "undefined" in reason


def test_the_ranking_can_be_empty_and_that_is_the_finding():
    """Under a tight enough bound nothing qualifies, and saying so is more
    useful than naming the least bad option."""
    with pytest.raises(SystemExit, match="undefined"):
        _decide(slo_bound_ms=5, prompt_tokens=8192, batch=32)


def test_a_prior_carries_an_unbounded_band_not_a_confident_one():
    """A prior has been observed above 40 percent error in this corpus. A
    decision resting on one must not look as certain as a measured cell."""
    r = _decide()
    priors = [c for c in r.candidates if not c["measured"]]
    assert priors, "the fleet should contain unmeasured silicon"
    assert all(c["band"] == PRIOR_BAND for c in priors)
    measured = [c for c in r.candidates if c["measured"]]
    assert all(c["band"] < PRIOR_BAND for c in measured)


def test_the_band_on_a_comparison_is_the_wider_of_the_two():
    """A difference is only as certain as its least certain side."""
    r = _decide()
    rec = next(c for c in r.candidates if c["silicon"] == r.recommended)
    inc = next(c for c in r.candidates + r.excluded
               if c["silicon"] == r.incumbent)
    assert r.band == max(rec["band"], inc["band"])


def test_a_margin_inside_the_band_does_not_clear_it():
    """The gate the agent reads. A recommendation inside the noise floor is
    not a recommendation."""
    r = _decide()
    assert r.clears_band == (r.margin > r.band)


def test_memory_that_will_not_hold_the_cache_is_excluded_with_a_reason():
    """The server would preempt and recompute, and recompute time appears in
    no term of the model. Estimating through it predicts a number for a
    regime the model does not describe."""
    r = _decide(batch=32, prompt_tokens=7680, slo_bound_ms=100_000)
    reasons = " ".join(c["excluded_reason"] or "" for c in r.excluded)
    assert "cache" in reasons or "memory" in reasons


def test_the_record_converts_to_what_the_agent_consumes():
    r = _decide()
    d = r.to_decision()
    assert d.recommended == r.recommended
    assert d.incumbent == r.incumbent
    assert d.confidence_band == r.band
    assert d.clears_band == r.clears_band


def test_provenance_is_on_every_class_of_figure():
    r = _decide()
    for key in ("cost", "band", "prices", "feasibility"):
        assert key in r.provenance


def test_an_unknown_incumbent_is_refused_rather_than_guessed():
    with pytest.raises(SystemExit, match="not in the fleet"):
        _decide(incumbent="a-card-that-does-not-exist")


def test_render_marks_the_recommendation_and_the_incumbent():
    r = _decide()
    text = render(r)
    assert "->" in text and "= " in text
    assert "MEASURED" in text and "prior" in text
