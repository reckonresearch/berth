"""Tests for the watchers and the GitHub integration.

Both are tested against fixtures rather than the network. A watcher that can
only be exercised against a live API is a watcher that gets exercised rarely,
and an integration that can only be tested by opening a real pull request is
one nobody runs before shipping.
"""

import pytest

from berth.agent import Trigger
from berth.github import (
    GitHubClient,
    GitHubError,
    RefusedByPolicy,
    RepoTarget,
    apply_edit,
    open_proposal,
)
from berth.watch import (
    WatchError,
    WatchState,
    build_detector,
    model_revision,
    watch_corpus,
    watch_models,
    watch_prices,
)

# ------------------------------------------------------------------ watchers

def _hf(sha):
    return {"sha": sha, "lastModified": "2026-08-01T00:00:00Z"}


def test_the_first_sight_of_a_model_fires_nothing():
    """We do not know whether it changed, so claiming it did would fire the
    agent on every new class the first time it is watched."""
    st = WatchState()
    fired = watch_models(["org/model"], st, fetch=lambda u: _hf("abc"))
    assert fired == []
    assert st.model_versions["org/model"] == "abc"


def test_a_moved_revision_fires_once_and_then_stops():
    st = WatchState()
    watch_models(["org/model"], st, fetch=lambda u: _hf("abc"))
    fired = watch_models(["org/model"], st, fetch=lambda u: _hf("def"))
    assert fired == [("org/model", "abc", "def")]
    again = watch_models(["org/model"], st, fetch=lambda u: _hf("def"))
    assert again == [], "a revision that has not moved is not an event"


def test_the_commit_is_watched_not_the_timestamp():
    """lastModified changes when the model card is edited. A README fix is not
    a placement event."""
    assert model_revision({"sha": "abc", "lastModified": "z"}) == "abc"


def test_an_unreachable_registry_is_not_a_change():
    """Recording an outage as a change would fire the agent on downtime."""
    st = WatchState()
    watch_models(["org/model"], st, fetch=lambda u: _hf("abc"))

    def down(_url):
        raise WatchError("503")

    assert watch_models(["org/model"], st, fetch=down) == []
    assert st.model_versions["org/model"] == "abc", "state must not be clobbered"


def test_a_price_moving_inside_the_epsilon_is_noise():
    """Providers adjust rates by fractions of a percent constantly and none of
    it changes a placement."""
    st = WatchState()
    watch_prices(lambda: {"h100-pcie": 2.60}, st)
    assert watch_prices(lambda: {"h100-pcie": 2.62}, st) == []
    moved = watch_prices(lambda: {"h100-pcie": 1.90}, st)
    assert moved and moved[0][0] == "h100-pcie"


def test_a_price_source_can_be_the_customers_own_rates():
    """Their contracted rate decides their placement, and it is usually not
    the published one."""
    st = WatchState()
    watch_prices(lambda: {"h100-pcie": 1.80}, st)     # negotiated, not list
    assert st.prices["h100-pcie"] == 1.80


def test_a_new_corpus_cell_fires_once():
    """A prior becoming a measurement is the only trigger where the answer can
    change without anything in the customer's world changing."""
    st = WatchState()
    watch_corpus([("l40s", "llama3-8b")], st)
    new = watch_corpus([("l40s", "llama3-8b"), ("mi300x", "llama3-8b")], st)
    assert new == [("mi300x", "llama3-8b")]
    assert watch_corpus([("l40s", "llama3-8b"), ("mi300x", "llama3-8b")], st) == []


def test_the_detector_polls_once_for_many_classes():
    """Polling per class hits a registry once per class watching the same
    model, which is wasteful and a good way to be rate limited."""
    calls = []

    def fetch(url):
        calls.append(url)
        return _hf("v2")

    st = WatchState(model_versions={"org/model": "v1"})
    detect = build_detector(st, fetch=fetch)
    detect.poll(["org/model"])
    assert len(calls) == 1

    class W:
        model_id = "org/model"
        model_key = "llama3-8b"
        current_silicon = "h100-pcie"

    assert Trigger.MODEL_VERSION in detect(W())
    assert Trigger.MODEL_VERSION in detect(W()), "cached, not re-polled"
    assert len(calls) == 1


# ------------------------------------------------------------------- github

def _client(log):
    def fake(method, url, token, body=None, opener=None):
        log.append((method, url, body))
        if url.endswith("/git/ref/heads/main"):
            return {"object": {"sha": "headsha"}}
        if "/contents/" in url and method == "GET":
            import base64
            content = "service: voice\n  accelerator: h100-pcie\n  replicas: 4\n"
            return {"content": base64.b64encode(content.encode()).decode(),
                    "sha": "filesha"}
        return {"number": 42, "html_url": "https://github.com/a/b/pull/42"}

    return GitHubClient("tok", request=fake)


def _repo(**kw):
    base = dict(owner="acme", name="infra", default_branch="main",
                allowed_paths=("deploy/voice.yaml",))
    base.update(kw)
    return RepoTarget(**base)


def test_a_path_outside_the_declaration_is_refused():
    """An agent that can write anywhere is an agent nobody grants access to."""
    with pytest.raises(RefusedByPolicy, match="not among the declared paths"):
        _repo().check("deploy/secrets.yaml")


def test_no_declared_paths_means_none():
    with pytest.raises(RefusedByPolicy, match="means none"):
        RepoTarget("a", "b").check("anything.yaml")


def test_committing_to_the_default_branch_is_refused():
    log = []
    c = _client(log)
    with pytest.raises(RefusedByPolicy, match="default branch"):
        c.create_branch(_repo(), "main", "sha")
    with pytest.raises(RefusedByPolicy, match="default branch"):
        c.put_file(_repo(), "deploy/voice.yaml", "x", branch="main", message="m")


def test_merging_is_refused_explicitly():
    """Present so the refusal is explicit rather than an absence. The agent
    proposes and the customer disposes."""
    with pytest.raises(RefusedByPolicy, match="does not merge"):
        _client([]).merge()


def test_a_stale_diff_is_refused_rather_than_guessed():
    """A configuration that no longer matches means the estimate is stale, and
    re-estimating is the correct response. Guessing at a deployment file is
    how access gets revoked."""
    original = "  accelerator: l40s\n"
    with pytest.raises(RefusedByPolicy, match="found 0"):
        apply_edit(original, "accelerator: h100-pcie", "accelerator: mi300x")


def test_an_ambiguous_edit_is_refused():
    original = "a:\n  accelerator: h100-pcie\nb:\n  accelerator: h100-pcie\n"
    with pytest.raises(RefusedByPolicy, match="found 2"):
        apply_edit(original, "accelerator: h100-pcie", "accelerator: l40s")


def test_a_single_match_is_edited_and_indentation_survives():
    original = "service: voice\n  accelerator: h100-pcie\n  replicas: 4\n"
    out = apply_edit(original, "accelerator: h100-pcie", "accelerator: mi300x")
    assert "  accelerator: mi300x\n" in out
    assert "replicas: 4" in out
    assert out.endswith("\n")


def test_open_proposal_reads_branches_writes_and_opens_in_that_order():
    from berth.agent import Decision, Trigger, WatchedClass, build_proposal
    log = []
    c = _client(log)
    w = WatchedClass("voice", "org/m", "llama3-8b", "h100-pcie", "3.1",
                     "p99_ttft_ms", 800, 8, 512, 128, "deploy/voice.yaml",
                     "acme/infra")
    d = Decision("mi300x", "h100-pcie", 0.469, 0.919, 0.106, 2, 0, True)
    p = build_proposal(w, Trigger.MODEL_VERSION, d, config_diff="x",
                       traces_url="https://t")
    r = open_proposal(c, _repo(), p)
    methods = [m for m, _u, _b in log]
    assert methods == ["GET", "GET", "POST", "PUT", "POST"]
    assert r["number"] == 42
    # The branch never collides with the default and names what it does.
    branch_body = next(b for m, u, b in log if u.endswith("/git/refs"))
    assert branch_body["ref"].startswith("refs/heads/berth/placement/")
    assert "main" not in branch_body["ref"].split("/")[-1]


def test_open_proposal_refuses_before_touching_anything():
    from berth.agent import Decision, Trigger, WatchedClass, build_proposal
    log = []
    c = _client(log)
    w = WatchedClass("voice", "org/m", "llama3-8b", "h100-pcie", "3.1",
                     "p99_ttft_ms", 800, 8, 512, 128, "deploy/OTHER.yaml",
                     "acme/infra")
    d = Decision("mi300x", "h100-pcie", 0.469, 0.919, 0.106, 2, 0, True)
    p = build_proposal(w, Trigger.MODEL_VERSION, d, config_diff="x",
                       traces_url="https://t")
    with pytest.raises(RefusedByPolicy):
        open_proposal(c, _repo(), p)
    assert log == [], "nothing may be called before the policy check"


def test_a_missing_token_is_refused_at_construction():
    with pytest.raises(GitHubError, match="token"):
        GitHubClient("")


# ------------------------------------------------- serving stack versions

def test_a_stack_release_fires_once():
    """The trigger with the most evidence behind it. On one H100 SXM, same
    card and model, vLLM 0.5.5 to 0.25 was worth 1.48x at batch 1 and 2.70x at
    batch 32, which is larger than most placement decisions this system
    makes."""
    from berth.watch import watch_serving_stacks
    st = WatchState()
    watch_serving_stacks(["vllm-project/vllm"], st,
                         fetch=lambda _u: {"tag_name": "v0.25.0"})
    fired = watch_serving_stacks(["vllm-project/vllm"], st,
                                 fetch=lambda _u: {"tag_name": "v0.26.0"})
    assert fired == [("vllm-project/vllm", "v0.25.0", "v0.26.0")]
    assert watch_serving_stacks(["vllm-project/vllm"], st,
                                fetch=lambda _u: {"tag_name": "v0.26.0"}) == []


def test_a_stack_release_touches_every_class():
    """The effect is on the server rather than on any one workload, so every
    class watching that stack is re-estimated."""
    from berth.agent import Trigger
    st = WatchState(stack_versions={"vllm-project/vllm": "v0.25.0"})
    detect = build_detector(st, stack_repos=["vllm-project/vllm"],
                            fetch=lambda _u: {"tag_name": "v0.26.0"})
    detect.poll([])

    class W:
        model_id = "unrelated/model"
        model_key = "llama3-8b"
        current_silicon = "l40s"

    assert Trigger.STACK_VERSION in detect(W())


def test_an_unreleased_repo_is_not_a_change():
    from berth.watch import watch_serving_stacks
    st = WatchState()
    assert watch_serving_stacks(["x/y"], st, fetch=lambda _u: {}) == []
