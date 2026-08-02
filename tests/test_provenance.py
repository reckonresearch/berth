"""Provenance of traces, and the prefill-floor fitter.

The corpus is the asset and its only guarantee is that every cell in it was
observed on hardware. These tests exist so that guarantee is enforced by code
rather than by care.
"""

import json

import pytest

from bench.check_contributed import check_file
from bench.fit_overhead import fit_floor
from bench.sounding import SCHEMA_VERSION, load_jsonl, provenance_of, save_jsonl
from berth.traces import TraceRecord


def rec(**kw):
    base = dict(silicon="h100-pcie", model_name="llama3-8b", batch=1,
                avg_prompt_tokens=512, avg_output_tokens=128,
                measured_ttft_ms=90.0, measured_tpot_ms=10.0)
    base.update(kw)
    return TraceRecord(**base)


# -- the field itself -------------------------------------------------------

def test_source_defaults_to_measured():
    assert rec().source == "measured"


def test_unknown_source_is_fatal():
    with pytest.raises(ValueError):
        rec(source="synthetic")


def test_generated_traces_are_labelled_mock():
    """generate_traces runs a hidden fleet plus noise. It is simulation, and
    calling its output measured would be the one unrecoverable error."""
    from berth.silicon import FLEET
    from berth.traces import generate_traces, make_true_fleet
    true = make_true_fleet(FLEET, {"h100-pcie": (0.45, 0.72)})
    traces = generate_traces(true, n_per_silicon=4, seed=1)
    assert traces and all(t.source == "mock" for t in traces)


# -- round trip and back-compatibility --------------------------------------

def test_roundtrip_preserves_source(tmp_path):
    p = tmp_path / "t.jsonl"
    save_jsonl([rec(source="mock"), rec(source="mock")], str(p))
    assert all(t.source == "mock" for t in load_jsonl(str(p)))
    assert json.loads(p.read_text().splitlines()[0])["schema"] == SCHEMA_VERSION


def test_schema_1_file_reads_as_measured(tmp_path):
    """The 60 P0 traces predate the field and are real. Rejecting them would
    be worse than assuming them; contributions are held to a stricter rule."""
    p = tmp_path / "old.jsonl"
    p.write_text(json.dumps({
        "schema": 1, "silicon": "l40s", "model_name": "llama3-8b", "batch": 1,
        "avg_prompt_tokens": 512, "avg_output_tokens": 128,
        "measured_ttft_ms": 120.0, "measured_tpot_ms": 22.0, "t": 0.0}) + "\n")
    assert load_jsonl(str(p))[0].source == "measured"


def test_future_schema_fails_loud(tmp_path):
    p = tmp_path / "future.jsonl"
    p.write_text('{"schema": 99, "silicon": "l40s", "model_name": "llama3-8b", '
                 '"batch": 1, "avg_prompt_tokens": 512, "avg_output_tokens": 128, '
                 '"measured_ttft_ms": 1.0, "measured_tpot_ms": 1.0}\n')
    with pytest.raises(ValueError):
        load_jsonl(str(p))


# -- the mixing rule --------------------------------------------------------

def test_provenance_of_homogeneous_set():
    assert provenance_of([rec(), rec()]) == "measured"
    assert provenance_of([rec(source="mock")]) == "mock"


def test_mixed_provenance_is_refused():
    with pytest.raises(SystemExit):
        provenance_of([rec(), rec(source="mock")])


# -- the contribution gate --------------------------------------------------

def write(tmp_path, name, records):
    p = tmp_path / name
    p.write_text("".join(json.dumps(r) + "\n" for r in records))
    return str(p)


BODY = dict(silicon="l40s", model_name="llama3-8b", batch=1,
            avg_prompt_tokens=512, avg_output_tokens=128,
            measured_ttft_ms=120.0, measured_tpot_ms=22.0)


def test_contribution_gate_accepts_measured(tmp_path):
    f = write(tmp_path, "good.jsonl", [{"schema": 2, "source": "measured", **BODY}])
    assert check_file(f) == []


def test_contribution_gate_rejects_mock(tmp_path):
    f = write(tmp_path, "mock.jsonl", [{"schema": 2, "source": "mock", **BODY}])
    assert any("mock" in m for _, m in check_file(f))


def test_contribution_gate_rejects_unlabelled(tmp_path):
    """Stricter than the loader on purpose: we know our own v1 files are
    hardware, we do not know a stranger's."""
    f = write(tmp_path, "old.jsonl", [{"schema": 1, **BODY}])
    problems = check_file(f)
    assert any("source" in m for _, m in problems)
    assert any("v2" in m for _, m in problems)


# -- the floor fitter -------------------------------------------------------

def synth_floor_traces(floor_ms, slope_ms_per_tok, lengths, batch=1):
    return [rec(batch=batch, avg_prompt_tokens=L,
                measured_ttft_ms=floor_ms + slope_ms_per_tok * L)
            for L in lengths]


def test_fit_recovers_a_known_floor():
    traces = synth_floor_traces(54.6, 0.05, [512, 2048, 8192])
    fit = fit_floor(traces)["h100-pcie"]
    assert abs(fit["floor_ms"] - 54.6) < 0.5
    assert abs(fit["slope_ms_per_token"] - 0.05) < 1e-3


def test_fit_ignores_batched_cells():
    """Batched cells carry the serial-admission term; including them would
    inflate the intercept and hand back a floor that is really contention."""
    good = synth_floor_traces(54.6, 0.05, [512, 2048, 8192])
    noise = synth_floor_traces(4000.0, 0.05, [512, 2048, 8192], batch=32)
    assert abs(fit_floor(good + noise)["h100-pcie"]["floor_ms"] - 54.6) < 0.5


def test_fit_refuses_a_flat_sweep():
    traces = synth_floor_traces(54.6, 0.05, [2000, 2050, 2100])
    assert "error" in fit_floor(traces)["h100-pcie"]


def test_fit_refuses_too_few_lengths():
    traces = synth_floor_traces(54.6, 0.05, [512, 8192])
    assert "error" in fit_floor(traces)["h100-pcie"]


def test_fit_flags_a_negative_floor():
    traces = synth_floor_traces(-30.0, 0.05, [512, 2048, 8192])
    assert fit_floor(traces)["h100-pcie"]["negative"] is True
