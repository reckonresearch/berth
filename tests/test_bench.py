"""Smoke test: the P0 pipeline (mock sweep -> JSONL -> calibrate) end to end."""
from types import SimpleNamespace

from bench.run_sweep import DEFAULT_GRID, load_jsonl, run_sweep, save_jsonl
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

    from bench.run_sweep import SCHEMA_VERSION
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
