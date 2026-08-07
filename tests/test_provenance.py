"""Provenance of traces, and the prefill-floor fitter.

The corpus is the asset and its only guarantee is that every cell in it was
observed on hardware. These tests exist so that guarantee is enforced by code
rather than by care.
"""

import json
from unittest import mock

import pytest

from bench.check_contributed import check_file
from bench.fit_overhead import fit_floor
from bench.sounding import (
    SCHEMA_VERSION,
    load_jsonl,
    provenance_of,
    save_jsonl,
    verify_served_model,
)
from berth.estimate import KV_PRESSURE_WARN, estimate
from berth.silicon import FLEET
from berth.traces import TraceRecord
from berth.workload import MODELS, WorkloadSpec, profile


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
    assert check_file(f)[0] == []


def test_contribution_gate_rejects_mock(tmp_path):
    f = write(tmp_path, "mock.jsonl", [{"schema": 2, "source": "mock", **BODY}])
    assert any("mock" in m for _, m in check_file(f)[0])


def test_contribution_gate_rejects_unlabelled(tmp_path):
    """Stricter than the loader on purpose: we know our own v1 files are
    hardware, we do not know a stranger's."""
    f = write(tmp_path, "old.jsonl", [{"schema": 1, **BODY}])
    problems, _ = check_file(f)
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


# -- the served-model guard -------------------------------------------------

def _fake_endpoint(served_ids, status=200):
    """Stand in for /v1/models without opening a socket."""

    class Resp:
        def __init__(self):
            self.status = status
        def read(self):
            return json.dumps({"data": [{"id": i} for i in served_ids]}).encode()

    conn = mock.MagicMock()
    conn.getresponse.return_value = Resp()
    return mock.patch("http.client.HTTPConnection", return_value=conn)


def test_guard_accepts_a_matching_model():
    with _fake_endpoint(["meta-llama/Meta-Llama-3-8B"]):
        assert verify_served_model("http://x:8000",
                                   "meta-llama/Meta-Llama-3-8B") == "meta-llama/Meta-Llama-3-8B"


def test_guard_refuses_a_different_model():
    """The silent-corruption case: the sweep would succeed and every cell would
    be attributed to weights the server never ran."""
    with _fake_endpoint(["Qwen/Qwen2.5-7B-Instruct"]):
        with pytest.raises(SystemExit) as e:
            verify_served_model("http://x:8000", "meta-llama/Meta-Llama-3-8B")
        assert "Qwen" in str(e.value)


def test_guard_refuses_when_the_endpoint_is_unreachable():
    from unittest import mock

    from bench.sounding import verify_served_model
    with mock.patch("http.client.HTTPConnection", side_effect=OSError("refused")):
        with pytest.raises(SystemExit):
            verify_served_model("http://x:8000", "any/model")


def test_guard_refuses_on_non_200():
    with _fake_endpoint([], status=404):
        with pytest.raises(SystemExit):
            verify_served_model("http://x:8000", "any/model")


# -- authentication ---------------------------------------------------------

def test_auth_headers_omit_the_token_when_absent():
    from bench.sounding import _auth_headers
    assert _auth_headers(None) == {"Content-Type": "application/json"}


def test_auth_headers_carry_a_bearer_token():
    from bench.sounding import _auth_headers
    h = _auth_headers("sk-abc")
    assert h["Authorization"] == "Bearer sk-abc"


def test_model_probe_refuses_on_401_with_a_key_hint():
    """A 401 must name the fix. The tester's response to an auth failure was
    to redeploy vLLM without auth, which measures a different server."""
    with _fake_endpoint([], status=401):
        with pytest.raises(SystemExit) as e:
            verify_served_model("http://x:8000", "any/model")
        assert "api-key" in str(e.value).lower()


# -- silicon provenance -----------------------------------------------------

def test_silicon_provenance_defaults_to_self_reported():
    assert rec().silicon_provenance == "self_reported"


def test_unknown_silicon_provenance_is_fatal():
    with pytest.raises(ValueError):
        rec(silicon_provenance="probably_an_h100")


def test_schema_2_file_loads_as_self_reported(tmp_path):
    """Pre-v3 traces were labelled by hand. self_reported describes that
    accurately; it is not a downgrade."""
    p = tmp_path / "v2.jsonl"
    p.write_text(json.dumps({
        "schema": 2, "source": "measured", "silicon": "l40s",
        "model_name": "llama3-8b", "batch": 1, "avg_prompt_tokens": 512,
        "avg_output_tokens": 128, "measured_ttft_ms": 120.0,
        "measured_tpot_ms": 22.0}) + "\n")
    assert load_jsonl(str(p))[0].silicon_provenance == "self_reported"


def test_mismatch_between_declared_and_detected_silicon_refuses():
    """The case the field exists for: real timings, wrong hardware label."""
    from bench.sounding import resolve_silicon_provenance
    with mock.patch("bench.sounding.detect_silicon",
                    return_value=("l40s", "NVIDIA L40S")):
        with pytest.raises(SystemExit) as e:
            resolve_silicon_provenance("h100-pcie", "http://localhost:8000")
        assert "l40s" in str(e.value)


def test_agreement_yields_captured():
    from bench.sounding import resolve_silicon_provenance
    with mock.patch("bench.sounding.detect_silicon",
                    return_value=("l40s", "NVIDIA L40S")):
        assert resolve_silicon_provenance("l40s", "http://localhost:8000") == "captured"


def test_remote_server_cannot_be_captured():
    """A remote endpoint does not report its hardware, so the honest answer is
    self_reported rather than a guess."""
    from bench.sounding import resolve_silicon_provenance
    assert resolve_silicon_provenance("l40s", "http://10.0.0.5:8000") == "self_reported"


def test_unrecognised_card_is_not_guessed():
    """An RTX PRO 6000 is not in the registry. Recording self_reported is
    correct; mapping it to the nearest fleet key would be a fabrication."""
    from bench.sounding import resolve_silicon_provenance
    with mock.patch("bench.sounding.detect_silicon",
                    return_value=(None, "NVIDIA RTX PRO 6000 Blackwell WS")):
        assert resolve_silicon_provenance("h100-pcie", "http://localhost:8000") == "self_reported"



# -- KV pressure ------------------------------------------------------------

def test_kv_pressure_flags_the_cell_that_thrashes():
    """L40S, Llama-3-8B, batch 32 at 7680 tokens needs more KV than one card
    has free. That is the cell carrying the batch-32 residual, and the
    estimate should say so rather than quietly assume a second GPU."""
    sig = profile(WorkloadSpec(model=MODELS["llama3-8b"], avg_prompt_tokens=7680,
                               avg_output_tokens=128, target_batch=32))
    e = estimate(sig, FLEET["l40s"], FLEET["l40s"].base_price_hr)
    assert e.kv_pressure > KV_PRESSURE_WARN


def test_kv_pressure_clear_on_a_larger_card():
    """Same configuration on an H100 PCIe has headroom, which is why its
    batch-32 residual is smaller."""
    sig = profile(WorkloadSpec(model=MODELS["llama3-8b"], avg_prompt_tokens=7680,
                               avg_output_tokens=128, target_batch=32))
    e = estimate(sig, FLEET["h100-pcie"], FLEET["h100-pcie"].base_price_hr)
    assert e.kv_pressure < KV_PRESSURE_WARN


def test_kv_pressure_is_single_device():
    """Reported per card, not per chosen layout. The layout math adds devices
    until the cache fits, which hides the answer to 'can this run on one'."""
    small = profile(WorkloadSpec(model=MODELS["llama3-8b"], avg_prompt_tokens=512,
                                 avg_output_tokens=16, target_batch=1))
    big = profile(WorkloadSpec(model=MODELS["llama3-8b"], avg_prompt_tokens=7680,
                               avg_output_tokens=128, target_batch=32))
    hw = FLEET["l40s"]
    assert (estimate(big, hw, hw.base_price_hr).kv_pressure
            > estimate(small, hw, hw.base_price_hr).kv_pressure * 10)
