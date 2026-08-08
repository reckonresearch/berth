"""Tests for the placement agent.

The agent's value is entirely in what it does not propose. An agent that opens
a pull request on every trigger gets muted, and a muted agent is worse than
none because it looks like coverage. Most of these tests are about silence.
"""


from berth.agent import (
    AgentRun,
    Decision,
    Trigger,
    WatchedClass,
    build_proposal,
    default_diff,
    evaluate,
    run,
)


def _watched(**kw):
    base = dict(workload_class="voice-agent-prod",
                model_id="NousResearch/Meta-Llama-3-8B", model_key="llama3-8b",
                current_silicon="h100-pcie", current_model_version="3.1",
                slo_metric="p99_ttft_ms", slo_bound_ms=800, batch=8,
                prompt_tokens=512, output_tokens=128,
                config_path="deploy/voice.yaml", repo="acme/infra")
    base.update(kw)
    return WatchedClass(**base)


def _decision(**kw):
    base = dict(recommended="l40s", incumbent="h100-pcie",
                recommended_cost=0.320, incumbent_cost=0.505,
                confidence_band=0.106, measured_cells=2, prior_cells=0,
                feasible=True)
    base.update(kw)
    return Decision(**base)


# ------------------------------------------------------------ when to speak

def test_a_change_clearing_the_band_is_proposed():
    propose, reason = evaluate(_watched(), Trigger.MODEL_VERSION, _decision())
    assert propose
    assert "improvement" in reason


def test_an_unchanged_answer_is_silent():
    """Most triggers should end here. A model version that does not move the
    placement is not news."""
    d = _decision(recommended="h100-pcie")
    propose, reason = evaluate(_watched(), Trigger.MODEL_VERSION, d)
    assert not propose
    assert reason == "the answer did not change"


def test_an_improvement_inside_the_band_is_silent():
    """A recommendation inside the noise floor is not a recommendation, and
    acting on one spends the customer's attention on a coin flip."""
    d = _decision(recommended_cost=0.480, incumbent_cost=0.505,
                  confidence_band=0.106)
    assert d.margin < d.confidence_band
    propose, reason = evaluate(_watched(), Trigger.PRICE_CHANGE, d)
    assert not propose
    assert "confidence band" in reason


def test_an_infeasible_incumbent_is_always_proposed():
    """A placement that cannot serve the workload is a problem the customer
    has whether or not the saving is large."""
    d = _decision(feasible=False, recommended_cost=0.500, incumbent_cost=0.505,
                  reason_infeasible="KV cache does not fit one card at batch 32")
    assert not d.clears_band
    propose, reason = evaluate(_watched(), Trigger.SLO_CHANGE, d)
    assert propose
    assert "cannot serve" in reason


def test_infeasible_with_no_alternative_is_silent():
    """Nothing to propose is not the same as nothing to say, but a pull
    request that changes nothing is noise. This is a capacity conversation,
    not a placement one."""
    d = _decision(feasible=False, recommended="h100-pcie",
                  reason_infeasible="no placement in the fleet fits")
    propose, reason = evaluate(_watched(), Trigger.SCHEDULED, d)
    assert not propose
    assert "capacity problem" in reason


# ------------------------------------------------------------ what it says

def test_the_body_states_when_a_recommendation_is_unmeasured():
    """A proposal resting entirely on spec-sheet arithmetic says so. A prior
    has been observed above 40 percent error in this corpus."""
    d = _decision(measured_cells=0, prior_cells=3)
    p = build_proposal(_watched(), Trigger.CORPUS_CELL, d,
                       config_diff="x", traces_url="https://example/traces")
    assert not d.rests_on_measurement
    assert "spec-sheet arithmetic" in p.body
    assert "40 percent" in p.body


def test_a_measured_recommendation_carries_no_warning():
    p = build_proposal(_watched(), Trigger.MODEL_VERSION, _decision(),
                       config_diff="x", traces_url="https://example/traces")
    assert "spec-sheet arithmetic" not in p.body
    assert "2 measured" in p.body


def test_the_body_carries_the_bound_and_the_workload_shape():
    """A reviewer should be able to disagree with the recommendation from the
    body alone, which requires knowing what it was optimised for."""
    p = build_proposal(_watched(), Trigger.PRICE_CHANGE, _decision(),
                       config_diff="x", traces_url="https://t")
    for fragment in ("p99_ttft_ms under 800 ms", "concurrency 8",
                     "512 prompt tokens", "128 output tokens"):
        assert fragment in p.body, fragment


def test_the_body_states_the_revert_path():
    """The first objection an infrastructure lead raises, answered in the
    proposal rather than in a follow-up."""
    p = build_proposal(_watched(), Trigger.MODEL_VERSION, _decision(),
                       config_diff="x", traces_url="https://t")
    assert "reverting the commit" in p.body
    assert "no traffic was read" in p.body.replace("\n", " ")


def test_the_diff_changes_one_field():
    """An agent that rewrites a deployment file is an agent nobody merges."""
    diff = default_diff(_watched(), _decision())
    body = [ln for ln in diff.splitlines()
            if not ln.startswith(("---", "+++", "@@"))]
    removed = [ln for ln in body if ln.startswith("-")]
    added = [ln for ln in body if ln.startswith("+")]
    assert len(removed) == 1 and len(added) == 1, body
    assert "deploy/voice.yaml" in diff


# ------------------------------------------------------------------ the loop

def test_run_records_suppressions_as_well_as_proposals():
    """A suppressed trigger is evidence about the trigger set, and the kill
    criterion is computed from it."""
    classes = [_watched(workload_class="a"), _watched(workload_class="b")]
    decisions = {"a": _decision(), "b": _decision(recommended="h100-pcie")}
    r = run(classes,
            resolve_decision=lambda w, t: decisions[w.workload_class],
            detect_triggers=lambda w: [Trigger.MODEL_VERSION])
    assert r.triggers_seen == 2
    assert len(r.proposals) == 1
    assert len(r.suppressed) == 1
    assert r.proposal_rate == 0.5


def test_proposal_rate_is_the_kill_criterion():
    """If fewer than one trigger in four produces a change clearing the band,
    the trigger set is wrong and the agent is noise."""
    r = AgentRun(triggers_seen=20)
    r.proposals = [None] * 3
    assert r.proposal_rate < 0.25, "this run would fail the kill criterion"
    r.proposals = [None] * 8
    assert r.proposal_rate >= 0.25


def test_no_triggers_means_no_estimates():
    """The agent does not re-estimate on a schedule it invented. Watching is
    cheap and estimating is not."""
    r = run([_watched()], resolve_decision=lambda w, t: _decision(),
            detect_triggers=lambda w: [])
    assert r.triggers_seen == 0 and r.estimates_run == 0 and not r.proposals


def test_a_custom_diff_renderer_is_used_when_given():
    """Deployment formats differ. The agent proposes a change to whatever the
    customer's configuration actually looks like."""
    r = run([_watched()], resolve_decision=lambda w, t: _decision(),
            detect_triggers=lambda w: [Trigger.CORPUS_CELL],
            render_diff=lambda w, d: f"custom for {w.workload_class}")
    assert r.proposals[0].config_diff == "custom for voice-agent-prod"


def test_margin_is_zero_when_the_incumbent_cost_is_unknown():
    """Divide-by-zero in a recommendation engine is a silent wrong answer."""
    d = _decision(incumbent_cost=0.0)
    assert d.margin == 0.0
    assert not d.clears_band
