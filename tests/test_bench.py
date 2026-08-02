"""Smoke test: the P0 pipeline (mock sweep -> JSONL -> calibrate) end to end."""
from types import SimpleNamespace

from bench.sounding import DEFAULT_GRID, load_jsonl, run_sweep, save_jsonl
from berth import FLEET
from berth.calibrate import calibrate


def test_mock_sweep_to_calibration(tmp_path):
    args = SimpleNamespace(mock=True, silicon="h100-sxm", model="llama3-8b",
                           seed=5, grid=DEFAULT_GRID, base_url=None, model_id=None)
    traces = run_sweep(args)
    assert len(traces) > 50
    p = tmp_path / "t.jsonl"
    save_jsonl(traces, str(p))
    loaded = load_jsonl(str(p))
    assert loaded == traces                      # lossless round-trip
    _, report = calibrate(FLEET, loaded)
    assert report.mape_calibrated < 0.10         # pipeline lands at noise floor


def test_physics_validation_passes_on_mock(tmp_path):
    """Term-by-term validator returns no FAILs when truth matches the model."""
    from bench.validate import check_bandwidth, check_kv_slope, check_prefill
    args = SimpleNamespace(mock=True, silicon="h100-sxm", model="llama3-8b",
                           seed=5, grid=DEFAULT_GRID, base_url=None, model_id=None)
    traces = run_sweep(args)
    for name, _, verdict in (check_bandwidth(traces, FLEET)
                             + check_kv_slope(traces, FLEET)
                             + check_prefill(traces, FLEET)):
        assert not str(verdict).startswith("FAIL"), (name, verdict)


def test_jsonl_schema_versioned(tmp_path):
    import json

    from bench.sounding import SCHEMA_VERSION
    args = SimpleNamespace(mock=True, silicon="h100-sxm", model="llama3-8b",
                           seed=5, grid=DEFAULT_GRID, base_url=None, model_id=None)
    traces = run_sweep(args)
    p = tmp_path / "t.jsonl"
    save_jsonl(traces, str(p))
    first = json.loads(open(p).readline())
    assert first["schema"] == SCHEMA_VERSION
    assert load_jsonl(str(p)) == traces
    # Unknown version fails loud, not silent.
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"schema": 99}\n')
    import pytest as _pt
    with _pt.raises(ValueError, match="schema v99"):
        load_jsonl(str(bad))


def test_v1_traces_load_as_bf16(tmp_path):
    # The 60-trace P0 files are schema v1 (no quant fields). They must keep
    # loading, defaulted to bf16 — not rejected by the v2 loader.
    p = tmp_path / "v1.jsonl"
    p.write_text('{"schema": 1, "silicon": "h100-pcie", "model_name": "llama3-8b", '
                 '"batch": 1, "avg_prompt_tokens": 512, "avg_output_tokens": 128, '
                 '"measured_ttft_ms": 100.0, "measured_tpot_ms": 10.0, "t": 0.0}\n')
    tr = load_jsonl(str(p))
    assert len(tr) == 1 and tr[0].w_bytes == 2.0 and tr[0].kv_bytes == 2.0


def test_fp8_trace_roundtrips_and_signature_is_quantized(tmp_path):
    # FLAG 2 end-to-end: a fp8 cell must persist its quant AND be inverted as fp8.
    from berth import MODELS
    from berth.traces import TraceRecord
    fp8 = TraceRecord(silicon="mi300x", model_name="deepseek-v3", batch=8,
                      avg_prompt_tokens=2048, avg_output_tokens=128,
                      measured_ttft_ms=200.0, measured_tpot_ms=12.0,
                      w_bytes=1.0, kv_bytes=2.0)
    p = tmp_path / "fp8.jsonl"
    save_jsonl([fp8], str(p))
    assert load_jsonl(str(p)) == [fp8]                       # quant survives round-trip
    # The reconstructed signature must carry fp8 weight bytes, not the bf16 default.
    sig = fp8.signature()
    assert sig.model.bytes_per_param == 1.0
    assert sig.model.weight_bytes == MODELS["deepseek-v3"].weight_bytes / 2


def test_rehearsal_report_end_to_end(tmp_path):
    """Full P0 pipeline dress rehearsal: traces -> calibrate -> physics checks
    (zero FAILs on synthetic truth) -> report generation."""
    from bench.report import _rehearsal_traces, build_report
    from bench.validate import check_bandwidth, check_kv_slope, check_prefill
    from berth import calibrate
    traces = _rehearsal_traces()
    fitted, _ = calibrate(FLEET, traces)
    checks = (check_bandwidth(traces, fitted) + check_kv_slope(traces, fitted)
              + check_prefill(traces, fitted))
    assert not any(str(v).startswith("FAIL") for _, _, v in checks)
    report = build_report(traces, rehearsal=True)
    assert "REHEARSAL" in report and "premium" in report
    assert report.count("|") > 40  # tables actually rendered
