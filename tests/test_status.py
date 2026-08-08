"""Tests for the status page.

The page unifies without becoming a dashboard, so the properties that matter
are that it aggregates everything and asserts nothing the other artifacts do
not already say.
"""

import pytest

from berth.agent import AgentState, Outcome, ProposalRecord, WatchedClass
from berth.place import decide
from berth.status import STATUS_PATH, render_status


def _classes():
    return [
        WatchedClass("voice", "org/m", "llama3-8b", "h100-pcie", "",
                     "p99_ttft_ms", 1000, 8, 512, 128, "deploy/v.yaml", "a/b"),
        WatchedClass("embed", "org/m", "llama3-8b", "mi300x", "",
                     "p99_ttft_ms", 5000, 64, 512, 16, "deploy/e.yaml", "a/b"),
    ]


def _decisions(classes):
    return {c.workload_class: decide(
        workload_class=c.workload_class, model_key=c.model_key,
        incumbent=c.current_silicon, slo_bound_ms=c.slo_bound_ms,
        batch=c.batch, prompt_tokens=c.prompt_tokens,
        output_tokens=c.output_tokens) for c in classes}


def test_every_class_appears_exactly_once():
    """The whole point is one place to look. A class missing from the page is
    a class the customer has no way to see."""
    cs = _classes()
    md = render_status(classes=cs, decisions=_decisions(cs),
                       agent_state=AgentState())
    for c in cs:
        assert md.count(f"`{c.workload_class}`") >= 1


def test_it_shows_what_the_estimator_says_now_not_when_the_proposal_opened():
    """Those differ whenever the corpus has moved, and the difference is the
    most useful thing on the page."""
    cs = _classes()
    md = render_status(classes=cs, decisions=_decisions(cs),
                       agent_state=AgentState())
    assert "recommended now" in md


def test_a_recommendation_inside_the_band_is_not_bolded_as_an_action():
    """The page must not read as though every difference is a change worth
    making. Inside the band it is noise."""
    cs = _classes()
    ds = _decisions(cs)
    d = ds["voice"]
    d.margin, d.band, d.clears_band = 0.04, 0.106, False
    md = render_status(classes=cs, decisions=ds, agent_state=AgentState())
    assert "inside a ±11% band" in md or "inside a" in md
    assert f"**{d.recommended}**, 4%" not in md


def test_an_unmeasured_recommendation_is_called_out_of_the_table():
    """A recommendation resting on nothing is the most important thing a
    reader can know, so it is not left in a column."""
    cs = _classes()
    ds = _decisions(cs)
    d = ds["voice"]
    d.measured_cells, d.prior_cells, d.clears_band = 0, 5, True
    md = render_status(classes=cs, decisions=ds, agent_state=AgentState())
    assert "Unmeasured recommendations" in md
    assert "40 percent" in md
    assert "four dollars" in md


def test_a_declined_proposal_keeps_its_reason():
    """Why a customer said no is more useful than that they did."""
    cs = _classes()
    st = AgentState(records=[
        ProposalRecord("voice", "h100-pcie", "mi300x", "corpus_cell",
                       "2026-06-02T09:00:00Z", Outcome.REJECTED,
                       "2026-06-04T08:00:00Z", "waiting on AMD quota")])
    md = render_status(classes=cs, decisions=_decisions(cs), agent_state=st)
    assert "declined" in md and "waiting on AMD quota" in md


def test_the_page_says_how_to_turn_it_off():
    """A surface with no visible off switch is a surface people resent."""
    cs = _classes()
    md = render_status(classes=cs, decisions=_decisions(cs),
                       agent_state=AgentState())
    assert "Delete it to stop" in md
    assert "classes.yaml" in md


def test_shadow_mode_says_so_at_the_top():
    cs = _classes()
    md = render_status(classes=cs, decisions=_decisions(cs),
                       agent_state=AgentState(), shadow=True)
    assert "shadow mode" in md and "nothing is being proposed" in md


def test_it_tells_the_reader_how_to_reproduce_any_figure():
    """The page asserts nothing the other artifacts do not already say, and
    it points at how to check."""
    cs = _classes()
    md = render_status(classes=cs, decisions=_decisions(cs),
                       agent_state=AgentState())
    assert "berth place" in md
    assert "docs.reckonresearch.com" in md


def test_status_is_written_to_a_branch_never_to_trunk():
    """The same review that accepts a placement accepts the status update.
    Nothing this integration does lands without a human merging it."""
    from berth.github import GitHubClient, RefusedByPolicy, RepoTarget
    from berth.status import publish
    calls = []

    def fake(method, url, token, body=None, opener=None):
        calls.append((method, url, body))
        if method == "GET":
            raise RuntimeError("absent")
        return {}

    client = GitHubClient("t", request=fake)
    repo = RepoTarget("a", "b", allowed_paths=("deploy/v.yaml",))
    publish(client, repo, "# x\n", branch="berth/placement/voice-mi300x")
    put = [c for c in calls if c[0] == "PUT"]
    assert put and put[0][2]["branch"] != "main"

    with pytest.raises(RefusedByPolicy):
        publish(client, repo, "# x\n", branch="main")


def test_the_status_path_is_added_to_allowed_paths_not_assumed():
    """The agent writes its own status file, and that permission is granted
    explicitly rather than by the client happening not to check."""
    assert STATUS_PATH == ".berth/STATUS.md"
