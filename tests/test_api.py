"""Tests for the placement API.

The API is where the control-plane boundary gets tested first, so most of
these are about what it refuses. A caller that can ask for a placement can
also try to ask for a completion, and the answer to that has to be a stated
refusal rather than a 404 they might read as a missing feature.
"""

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from berth.api import Handler


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 8931), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.3)
    yield "http://127.0.0.1:8931"
    srv.shutdown()


def post(base, path, body):
    r = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                               headers={"content-type": "application/json"})
    try:
        return 200, json.loads(urllib.request.urlopen(r, timeout=8).read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_it_returns_a_placement_decision(server):
    code, d = post(server, "/v1/place",
                   {"model": "llama3-8b", "slo_ms": 800, "concurrency": 8,
                    "prompt_tokens": 512, "output_tokens": 128})
    assert code == 200
    assert d["recommended"] and d["candidates"]
    assert "act" in d and "why" in d


def test_it_says_whether_to_act_not_just_what_ranked_first(server):
    """A caller acting on this automatically has no chance to ask a follow-up
    question, so the answer has to carry the verdict as well as the ranking."""
    _, d = post(server, "/v1/place",
                {"model": "llama3-8b", "slo_ms": 800, "concurrency": 8,
                 "prompt_tokens": 512, "output_tokens": 128})
    assert isinstance(d["act"], bool)
    assert d["act"] == (d["margin"] > d["band"])
    assert "band" in d["why"]


def test_it_refuses_to_be_an_inference_endpoint(server):
    """The boundary. A party that carries traffic cannot credibly rank the
    placements it carries traffic for, so this refuses with the reason rather
    than 404ing in a way that reads as a missing feature."""
    for path in ("/v1/completions", "/v1/chat/completions", "/v1/generate"):
        code, d = post(server, path, {"prompt": "hello"})
        assert code == 400, path
        assert "not an inference endpoint" in d["error"]
        assert "carries traffic" in d["detail"]
        assert d["see"].startswith("https://")


def test_every_number_carries_its_basis(server):
    """A caller cannot ask whether a figure came off hardware or a spec sheet,
    so the answer has to say."""
    _, d = post(server, "/v1/place",
                {"model": "llama3-8b", "slo_ms": 800, "concurrency": 8,
                 "prompt_tokens": 512, "output_tokens": 128})
    assert "measured_cells" in d and "prior_cells" in d
    assert all("measured" in c for c in d["candidates"])
    assert "band" in d["provenance"]


def test_no_feasible_placement_is_a_result_not_an_error(server):
    """Cost per served token where nothing meets the bound is undefined, not
    high. 422 rather than 500: the request was fine and the answer is that
    there isn't one."""
    code, d = post(server, "/v1/place",
                   {"model": "llama3-8b", "slo_ms": 5, "concurrency": 32,
                    "prompt_tokens": 8192, "output_tokens": 128})
    assert code == 422
    assert "no feasible placement" in d["error"]
    assert "undefined" in d["detail"]


def test_missing_fields_name_themselves(server):
    code, d = post(server, "/v1/place", {"model": "llama3-8b"})
    assert code == 400
    for f in ("slo_ms", "concurrency", "prompt_tokens", "output_tokens"):
        assert f in d["error"]


def test_an_unknown_model_lists_the_known_ones(server):
    code, d = post(server, "/v1/place",
                   {"model": "not-a-model", "slo_ms": 800, "concurrency": 8,
                    "prompt_tokens": 512, "output_tokens": 128})
    assert code == 400 and "llama3-8b" in d["error"]


def test_the_fleet_is_readable_and_says_what_is_measured(server):
    d = json.loads(urllib.request.urlopen(server + "/v1/silicon", timeout=5).read())
    assert d["silicon"] and d["models"]
    assert all("measured" in v for v in d["silicon"].values())
    assert sum(1 for v in d["silicon"].values() if v["measured"]) >= 5


def test_an_unknown_route_lists_the_real_ones(server):
    code, d = post(server, "/v1/nonsense", {})
    assert code == 404 and "/v1/place" in d["routes"]


def test_it_binds_to_loopback_by_default():
    """A placement API is an internal service. Binding to all interfaces by
    default is how something ends up on the public internet because nobody
    passed a flag."""
    import inspect

    from berth.api import serve
    assert inspect.signature(serve).parameters["host"].default == "127.0.0.1"


# -- the CLI paths a customer actually runs ---------------------------------

def test_pilot_reads_a_yaml_declaration(tmp_path):
    """cmd_pilot used json.load on a file that is YAML, so the command every
    customer would run to use pilot crashed on its first line. The declaration
    parser existed and the CLI never called it."""
    from berth.cli import build_parser
    d = tmp_path / "classes.yaml"
    d.write_text("""version: 1
repo:
  allowed_paths: [d.yaml]
classes:
  - name: v
    model_id: o/m
    model: llama3-8b
    running_on: h100-pcie
    config_path: d.yaml
    slo: {metric: p99_ttft_ms, bound_ms: 800}
    workload: {concurrency: 8, prompt_tokens: 512, output_tokens: 128}
""")
    args = build_parser().parse_args(["pilot", "--classes", str(d)])
    assert args.func(args) == 0


def test_live_without_a_token_refuses_rather_than_doing_nothing(tmp_path):
    """Without a credential this would decide and silently change nothing,
    which is worse than refusing: you would believe it ran."""
    import os

    import pytest

    from berth.cli import build_parser
    d = tmp_path / "c.yaml"
    d.write_text("""version: 1
repo:
  allowed_paths: [d.yaml]
classes:
  - name: v
    model_id: o/m
    model: llama3-8b
    running_on: h100-pcie
    config_path: d.yaml
    slo: {metric: p99_ttft_ms, bound_ms: 800}
    workload: {concurrency: 8, prompt_tokens: 512, output_tokens: 128}
""")
    old = os.environ.pop("GITHUB_TOKEN", None)
    try:
        args = build_parser().parse_args(
            ["pilot", "--classes", str(d), "--live", "--repo", "a/b"])
        with pytest.raises(SystemExit, match="GITHUB_TOKEN"):
            args.func(args)
    finally:
        if old:
            os.environ["GITHUB_TOKEN"] = old
