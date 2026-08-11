"""Tests for the in-repo declaration.

The declaration bounds what the agent may do. Every test here is about
refusing rather than accepting, because a declaration quietly filled in is an
agent operating under terms nobody wrote down.
"""

import pytest

from berth.declaration import DeclarationError, load_yaml, parse

GOOD = """
version: 1
repo:
  allowed_paths:
    - deploy/voice.yaml
classes:
  - name: voice
    model_id: org/model
    model: llama3-8b
    running_on: h100-pcie
    config_path: deploy/voice.yaml
    slo:
      metric: p99_ttft_ms
      bound_ms: 800
    workload:
      concurrency: 8
      prompt_tokens: 512
      output_tokens: 128
"""


def test_a_good_declaration_parses_without_pyyaml():
    """The core has no dependencies, and a customer should not need to install
    one to be read."""
    d = load_yaml(GOOD, repo="acme/infra")
    assert d.version == 1
    assert d.allowed_paths == ("deploy/voice.yaml",)
    c = d.classes[0]
    assert c.workload_class == "voice"
    assert c.slo_bound_ms == 800
    assert c.batch == 8 and c.prompt_tokens == 512 and c.output_tokens == 128
    assert c.repo == "acme/infra"


def test_empty_allowed_paths_means_none():
    """An agent that can write anywhere is one nobody grants access to."""
    with pytest.raises(DeclarationError, match="means none"):
        parse({"version": 1, "repo": {"allowed_paths": []},
               "classes": [{"name": "x"}]})


def test_a_class_cannot_name_a_path_outside_the_allowed_list():
    """Permission is granted deliberately, not inferred from use."""
    raw = {"version": 1, "repo": {"allowed_paths": ["a.yaml"]},
           "classes": [{"name": "v", "model_id": "o/m", "model": "llama3-8b",
                        "running_on": "l40s", "config_path": "b.yaml",
                        "slo": {"bound_ms": 800},
                        "workload": {"concurrency": 1, "prompt_tokens": 1,
                                     "output_tokens": 1}}]}
    with pytest.raises(DeclarationError, match="not in repo.allowed_paths"):
        parse(raw)


def test_a_missing_field_is_refused_not_defaulted():
    raw = {"version": 1, "repo": {"allowed_paths": ["a.yaml"]},
           "classes": [{"name": "v", "config_path": "a.yaml"}]}
    with pytest.raises(DeclarationError, match="missing 'model_id'"):
        parse(raw)


def test_an_unversioned_declaration_is_refused():
    """An unversioned file cannot be safely reinterpreted when the format
    changes."""
    with pytest.raises(DeclarationError, match="version must be 1"):
        parse({"repo": {"allowed_paths": ["a.yaml"]}, "classes": []})


def test_duplicate_class_names_are_refused():
    """A class name is the key the agent tracks state against, so two of the
    same would make proposals and their outcomes ambiguous."""
    one = {"name": "v", "model_id": "o/m", "model": "llama3-8b",
           "running_on": "l40s", "config_path": "a.yaml",
           "slo": {"bound_ms": 800},
           "workload": {"concurrency": 1, "prompt_tokens": 1, "output_tokens": 1}}
    with pytest.raises(DeclarationError, match="duplicate class names"):
        parse({"version": 1, "repo": {"allowed_paths": ["a.yaml"]},
               "classes": [one, dict(one)]})


def test_no_classes_is_refused_rather_than_silently_doing_nothing():
    with pytest.raises(DeclarationError, match="no classes declared"):
        parse({"version": 1, "repo": {"allowed_paths": ["a.yaml"]},
               "classes": []})


def test_the_declaration_produces_the_repo_target_the_client_enforces():
    """The same allowed paths bound the parser and the GitHub client, so a
    class cannot be declared that the client would then refuse."""
    d = load_yaml(GOOD, repo="acme/infra")
    t = d.repo_target("acme", "infra")
    assert t.allowed_paths == d.allowed_paths
    t.check("deploy/voice.yaml")
    from berth.github import RefusedByPolicy
    with pytest.raises(RefusedByPolicy):
        t.check("deploy/other.yaml")


# -- declared axes versus chosen axes ---------------------------------------

def test_pinning_a_provider_inside_a_class_is_refused():
    """Chips, providers and price bases are chosen by the engine. A constraint
    arriving as a declared axis makes a preference indistinguishable from a
    physical property."""
    raw = {"version": 1, "repo": {"allowed_paths": ["a.yaml"]},
           "classes": [{"name": "v", "model_id": "o/m", "model": "llama3-8b",
                        "running_on": "l40s", "config_path": "a.yaml",
                        "provider": "aws",
                        "slo": {"bound_ms": 800},
                        "workload": {"concurrency": 1, "prompt_tokens": 1,
                                     "output_tokens": 1}}]}
    with pytest.raises(DeclarationError, match="chosen by the engine"):
        parse(raw)


def test_a_narrowing_constraint_needs_a_reason():
    """A constraint costs money, and the reason is what lets anyone tell later
    whether it is still worth paying."""
    raw = {"version": 1, "repo": {"allowed_paths": ["a.yaml"]},
           "classes": [{"name": "v", "model_id": "o/m", "model": "llama3-8b",
                        "running_on": "l40s", "config_path": "a.yaml",
                        "constraints": {"providers": ["aws"]},
                        "slo": {"bound_ms": 800},
                        "workload": {"concurrency": 1, "prompt_tokens": 1,
                                     "output_tokens": 1}}]}
    with pytest.raises(DeclarationError, match="without a reason"):
        parse(raw)


def test_a_constraint_with_a_reason_parses_and_is_marked_as_narrowing():
    raw = {"version": 1, "repo": {"allowed_paths": ["a.yaml"]},
           "classes": [{"name": "v", "model_id": "o/m", "model": "llama3-8b",
                        "running_on": "l40s", "config_path": "a.yaml",
                        "constraints": {"providers": ["aws"],
                                        "reason": "data residency, EU only"},
                        "slo": {"bound_ms": 800},
                        "workload": {"concurrency": 1, "prompt_tokens": 1,
                                     "output_tokens": 1}}]}
    d = parse(raw)
    c = d.constraints["v"]
    assert c.providers == ("aws",) and c.narrows
    assert "residency" in c.reason


def test_on_demand_only_is_not_a_narrowing():
    """The default is not a constraint, so it needs no justification."""
    from berth.declaration import Constraints
    assert not Constraints().narrows
    assert Constraints(price_bases=("spot", "on-demand")).narrows


def test_an_unknown_price_basis_is_refused():
    raw = {"version": 1, "repo": {"allowed_paths": ["a.yaml"]},
           "classes": [{"name": "v", "model_id": "o/m", "model": "llama3-8b",
                        "running_on": "l40s", "config_path": "a.yaml",
                        "constraints": {"price_bases": ["preemptible"],
                                        "reason": "x"},
                        "slo": {"bound_ms": 800},
                        "workload": {"concurrency": 1, "prompt_tokens": 1,
                                     "output_tokens": 1}}]}
    with pytest.raises(DeclarationError, match="unknown price bases"):
        parse(raw)


def test_volume_is_declared_and_optional():
    """A percentage is not money until somebody says how much work there is.
    Absent means the class appears in the status page and not in the ledger,
    which is the honest treatment rather than a guessed volume."""
    raw = {"version": 1, "repo": {"allowed_paths": ["a.yaml"]},
           "classes": [{"name": "v", "model_id": "o/m", "model": "llama3-8b",
                        "running_on": "l40s", "config_path": "a.yaml",
                        "slo": {"bound_ms": 800},
                        "workload": {"concurrency": 1, "prompt_tokens": 1,
                                     "output_tokens": 1, "mtok_per_hour": 12.5}}]}
    assert parse(raw).classes[0].mtok_per_hour == 12.5
    del raw["classes"][0]["workload"]["mtok_per_hour"]
    assert parse(raw).classes[0].mtok_per_hour is None
