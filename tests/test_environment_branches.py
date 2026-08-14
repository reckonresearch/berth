"""Every branch selected by the environment, and a test that takes it.

Defect 15 was a fallback parser that could not read the format we publish. It
survived because every test ran in an environment where the fallback was never
taken. Fixing that one instance did not close the class: a branch chosen by
what happens to be installed is a branch nobody exercises until a customer has
the environment that chooses it.

This file enumerates them. A new `except ImportError` in the codebase without
a corresponding test here should be treated as an untested code path, and
`test_every_import_fallback_is_enumerated` fails when one appears.
"""

import builtins
import importlib
import sys
from pathlib import Path

# Every environment-selected branch, and where it is covered. A branch listed
# as `deleted` is one that should not exist; a branch listed as `untestable`
# needs a reason, because "we cannot test it" is a claim rather than a fact.
KNOWN_FALLBACKS = {
    ("berth/declaration.py", "yaml"): "test_declaration.py, three tests",
    ("bench/microbench.py", "torch"): "test_environment_branches.py, skip path",
    ("bench/microbench.py", "torch_xla"): "no accelerator in CI, skip path tested below",
    ("bench/microbench.py", "torch_neuronx"): "no accelerator in CI, skip path tested below",
}


# Standard library. An ImportError on these is a broken interpreter, not an
# environment-selected branch, and sweeping them in makes the register noise.
STDLIB = set(sys.stdlib_module_names)


def _import_fallbacks():
    """Find every third-party import guarded by `except ImportError`.

    Parsed with ast rather than regex: the first version matched every import
    inside a try block, including stdlib imports that happened to sit
    alongside a guarded one, which is a register full of things that cannot
    fail.
    """
    import ast
    found = set()
    root = Path(__file__).resolve().parent.parent
    for src in list((root / "berth").glob("*.py")) + list((root / "bench").glob("*.py")):
        rel = f"{src.parent.name}/{src.name}"
        for node in ast.walk(ast.parse(src.read_text())):
            if not isinstance(node, ast.Try):
                continue
            catches = any(
                (isinstance(h.type, ast.Name) and h.type.id == "ImportError")
                or (isinstance(h.type, ast.Tuple)
                    and any(isinstance(e, ast.Name) and e.id == "ImportError"
                            for e in h.type.elts))
                for h in node.handlers)
            if not catches:
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Import):
                    names = [a.name.split(".")[0] for a in inner.names]
                elif isinstance(inner, ast.ImportFrom) and inner.module:
                    names = [inner.module.split(".")[0]]
                else:
                    continue
                for n in names:
                    if n not in STDLIB and n != "berth" and n != "bench":
                        found.add((rel, n))
    return found


def test_every_import_fallback_is_enumerated():
    """A new environment-selected branch must be registered here.

    Not because registration is a test, but because the act of adding a line
    forces the question: what happens to a customer who takes this branch, and
    has anyone run it?
    """
    found = _import_fallbacks()
    unregistered = found - set(KNOWN_FALLBACKS)
    assert not unregistered, (
        f"environment-selected branches with no entry in KNOWN_FALLBACKS: "
        f"{sorted(unregistered)}. Each is a code path a customer can take and "
        f"nobody has run. Add it with where it is covered, or delete the "
        f"branch.")

    stale = set(KNOWN_FALLBACKS) - found
    assert not stale, (
        f"KNOWN_FALLBACKS names branches that no longer exist: "
        f"{sorted(stale)}. A register that drifts from the code is worse than "
        f"none, because it reads as coverage.")


def test_the_declaration_reader_works_without_pyyaml(monkeypatch):
    """The branch defect 15 lived in, taken deliberately."""
    real = builtins.__import__

    def blocked(name, *a, **k):
        if name == "yaml":
            raise ImportError("PyYAML deliberately absent")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    for mod in [m for m in sys.modules if m.startswith("yaml")]:
        del sys.modules[mod]

    decl = importlib.import_module("berth.declaration")
    d = decl.load_yaml("""version: 1
repo:
  allowed_paths: [deploy/voice.yaml]
classes:
  - name: voice
    model_id: org/model
    model: llama3-8b
    running_on: h100-pcie
    config_path: deploy/voice.yaml
    slo: {metric: p99_ttft_ms, bound_ms: 800}
    workload: {concurrency: 8, prompt_tokens: 512, output_tokens: 128}
""", repo="acme/infra")
    assert d.allowed_paths == ("deploy/voice.yaml",)
    assert d.classes[0].workload_class == "voice"


def test_the_microbenchmark_skips_rather_than_fabricating_a_ceiling():
    """With no accelerator visible it must decline, not guess.

    A microbenchmark that returns a number when it measured nothing is the
    worst possible failure here: the number enters the corpus labelled
    MEASURED and there is no way to tell afterwards.
    """
    import subprocess
    r = subprocess.run([sys.executable, "-m", "bench.microbench"],
                       capture_output=True, text=True, timeout=120)
    # Exit 2 is the declared "nothing to measure" code. Anything else on a
    # machine with no accelerator means it produced output it should not have.
    assert r.returncode == 2, (
        f"expected exit 2 with no accelerator, got {r.returncode}. "
        f"stdout: {r.stdout[:400]}")
    # Two legitimate reasons to decline, and the message must name which:
    # torch absent, or torch present with no device visible. An operator on
    # the wrong machine needs to know which one they are looking at.
    out = r.stdout.lower()
    assert ("torch is not installed" in out
            or "no supported accelerator" in out), r.stdout[:200]
    # And whichever it is, no number. A ceiling reported by a run that
    # measured nothing is the worst failure available here, because it enters
    # the corpus labelled MEASURED and cannot be distinguished afterwards.
    assert "tflops" not in out and "gb/s" not in out


def test_a_missing_optional_dependency_never_changes_a_measured_number():
    """The property that makes the rest of this file matter.

    An environment-selected branch is tolerable when it changes convenience.
    It is not tolerable when it changes a number, because the number carries a
    provenance label and the label does not record which branch produced it.
    """
    from berth.estimate import estimate
    from berth.silicon import FLEET
    from berth.workload import MODELS, WorkloadSpec, profile

    sig = profile(WorkloadSpec(model=MODELS["llama3-8b"], avg_prompt_tokens=512,
                               avg_output_tokens=128, target_batch=8))
    hw = FLEET["h100-pcie"]
    baseline = estimate(sig, hw, hw.base_price_hr)

    real = builtins.__import__

    def blocked(name, *a, **k):
        if name in ("yaml", "torch", "numpy"):
            raise ImportError(f"{name} deliberately absent")
        return real(name, *a, **k)

    try:
        builtins.__import__ = blocked
        again = estimate(sig, hw, hw.base_price_hr)
    finally:
        builtins.__import__ = real

    assert again.cost_per_mtok == baseline.cost_per_mtok
    assert again.ttft_ms == baseline.ttft_ms
    assert again.tpot_ms == baseline.tpot_ms
