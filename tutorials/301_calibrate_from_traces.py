"""Tutorial 301: calibrate efficiency factors from measured traces.

Uses the mock harness so it runs with zero GPUs; on real hardware only the
data source changes (see docs/bench.md).
Run: python tutorials/301_calibrate_from_traces.py
Expected: holdout MAPE at/near the noise floor; fits with 95% CIs.
"""
from types import SimpleNamespace

from bench.run_sweep import DEFAULT_GRID, run_sweep
from berth import FLEET, calibrate

args = SimpleNamespace(mock=True, silicon="h100-sxm", model="llama3-8b",
                       seed=7, grid=DEFAULT_GRID, base_url=None, model_id=None)
traces = run_sweep(args)
fleet, report = calibrate(FLEET, traces)
(mfu_lo, mfu_hi), (bw_lo, bw_hi) = report.ci95["h100-sxm"]
print(f"\nholdout MAPE: {report.mape_prior:.1%} -> {report.mape_calibrated:.1%}")
print(f"h100-sxm mfu    {fleet['h100-sxm'].mfu:.3f}  95% CI [{mfu_lo:.3f}, {mfu_hi:.3f}]")
print(f"h100-sxm bw_eff {fleet['h100-sxm'].bw_eff:.3f}  95% CI [{bw_lo:.3f}, {bw_hi:.3f}]")
