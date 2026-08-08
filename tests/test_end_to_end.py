"""The whole loop, without a network.

Registry moves, the estimator decides, the agent proposes, pull requests open.
Run as a test rather than as a script, because a demonstration that only lives
in a terminal history is a demonstration nobody re-runs.
"""

import base64
import json

import pytest

from berth.agent import AgentState, WatchedClass, run
from berth.github import GitHubClient, RefusedByPolicy, RepoTarget, open_proposal
from berth.place import decide
from berth.watch import WatchState, build_detector

FILES = {
    "deploy/voice.yaml": "service: voice\n  accelerator: h100-pcie\n  replicas: 4\n",
    "deploy/embed.yaml": "service: embed\n  accelerator: l40s\n  replicas: 8\n",
}


def _classes():
    return [
        WatchedClass("voice-agent-prod", "NousResearch/Meta-Llama-3-8B",
                     "llama3-8b", "h100-pcie", "abc123", "p99_ttft_ms", 1000,
                     8, 512, 128, "deploy/voice.yaml", "acme/infra"),
        WatchedClass("batch-embed", "NousResearch/Meta-Llama-3-8B",
                     "llama3-8b", "l40s", "abc123", "p99_ttft_ms", 5000,
                     64, 512, 16, "deploy/embed.yaml", "acme/infra"),
    ]


def _resolve(w, _trig):
    return decide(workload_class=w.workload_class, model_key=w.model_key,
                  incumbent=w.current_silicon, slo_bound_ms=w.slo_bound_ms,
                  batch=w.batch, prompt_tokens=w.prompt_tokens,
                  output_tokens=w.output_tokens).to_decision()


def _github(log):
    prs = []

    def fake(method, url, token, body=None, opener=None):
        path = url.replace("https://api.github.com", "")
        log.append((method, path, body))
        if path.endswith("/git/ref/heads/main"):
            return {"object": {"sha": "head"}}
        if "/contents/" in path and method == "GET":
            f = path.split("/contents/")[1].split("?")[0]
            return {"content": base64.b64encode(FILES[f].encode()).decode(),
                    "sha": "filesha"}
        if path.endswith("/pulls"):
            prs.append(body)
            return {"number": 118 + len(prs),
                    "html_url": f"https://github.com/acme/infra/pull/{117+len(prs)}"}
        return {}

    return GitHubClient("ghp_fake", request=fake), prs


def test_registry_move_becomes_a_pull_request():
    log = []
    client, prs = _github(log)
    wstate = WatchState(model_versions={"NousResearch/Meta-Llama-3-8B": "abc123"})
    detect = build_detector(wstate, fetch=lambda _u: {"sha": "def456"})
    detect.poll(["NousResearch/Meta-Llama-3-8B"])

    result = run(_classes(), _resolve, detect, state=AgentState())
    assert result.triggers_seen == 2
    assert result.proposals

    repo = RepoTarget("acme", "infra",
                      allowed_paths=tuple(FILES))
    for p in result.proposals:
        open_proposal(client, repo, p)

    methods = [m for m, _p, _b in log]
    assert "DELETE" not in methods
    assert not any("merge" in p for _m, p, _b in log), "the agent never merges"
    assert len(prs) == len(result.proposals)


def test_nothing_is_written_to_the_default_branch():
    log = []
    client, _prs = _github(log)
    wstate = WatchState(model_versions={"NousResearch/Meta-Llama-3-8B": "abc"})
    detect = build_detector(wstate, fetch=lambda _u: {"sha": "def"})
    detect.poll(["NousResearch/Meta-Llama-3-8B"])
    result = run(_classes(), _resolve, detect, state=AgentState())
    repo = RepoTarget("acme", "infra", allowed_paths=tuple(FILES))
    for p in result.proposals:
        open_proposal(client, repo, p)
    for _m, _p, body in log:
        if body and "branch" in body:
            assert body["branch"] != "main"
        if body and "ref" in body:
            assert not body["ref"].endswith("/main")


def test_an_undeclared_path_stops_the_loop_before_any_call():
    log = []
    client, _prs = _github(log)
    wstate = WatchState(model_versions={"NousResearch/Meta-Llama-3-8B": "abc"})
    detect = build_detector(wstate, fetch=lambda _u: {"sha": "def"})
    detect.poll(["NousResearch/Meta-Llama-3-8B"])
    result = run(_classes(), _resolve, detect, state=AgentState())
    narrow = RepoTarget("acme", "infra", allowed_paths=("something/else.yaml",))
    with pytest.raises(RefusedByPolicy):
        open_proposal(client, narrow, result.proposals[0])
    assert log == []


def test_the_second_pass_proposes_nothing_new():
    """The registry has not moved again, and the open proposals are known."""
    wstate = WatchState(model_versions={"NousResearch/Meta-Llama-3-8B": "abc"})
    astate = AgentState()
    detect = build_detector(wstate, fetch=lambda _u: {"sha": "def"})
    detect.poll(["NousResearch/Meta-Llama-3-8B"])
    first = run(_classes(), _resolve, detect, state=astate)
    assert first.proposals

    detect2 = build_detector(wstate, fetch=lambda _u: {"sha": "def"})
    detect2.poll(["NousResearch/Meta-Llama-3-8B"])
    second = run(_classes(), _resolve, detect2, state=astate)
    assert not second.proposals, "an unchanged world proposes nothing"


def test_the_cli_classes_file_shape_round_trips(tmp_path):
    """The shape `berth pilot --classes` expects, pinned so the docs and the
    parser cannot drift apart."""
    from dataclasses import asdict
    p = tmp_path / "classes.json"
    p.write_text(json.dumps([asdict(c) for c in _classes()]))
    loaded = [WatchedClass(**c) for c in json.loads(p.read_text())]
    assert loaded == _classes()
