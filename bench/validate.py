"""Physics validation: term-by-term checks on measured traces.

Aggregate MAPE can hide compensating errors. Each roofline term has a clean
single-variable test; this report runs all of them against a traces JSONL
and prints PASS/FAIL per term, then checks residual structure after
calibration (structured residuals = missing physics, not noise).

Checks:
  1. bandwidth  — batch-1 decode: TPOT ~= active weight bytes / eff. bandwidth
  2. kv-slope   — TPOT linear in context at fixed batch; slope matches
                  batch * kv_bytes_per_token / eff. bandwidth
  3. prefill    — TTFT linear in prompt length; slope -> mfu within bounds
  4. residuals  — post-calibration relative errors grouped by batch and by
                  context bucket; grouped means should sit near zero

Usage:
  python -m bench.validate traces.jsonl [more.jsonl ...]
      [--bw-ceiling GBps --flops-ceiling TFLOPS]   # from bench.microbench
"""

import argparse
import random
import statistics
import sys
from collections import defaultdict

from bench.run_sweep import load_jsonl
from berth.calibrate import calibrate
from berth.estimate import estimate, replica_layout
from berth.silicon import FLEET

TOL = 0.15  # per-term relative tolerance; matches the P0 pass criterion


def _theil_sen(xs, ys):
    """Median of pairwise slopes — robust slope for noisy, few-point data.
    OLS at 5% multiplicative noise over <=6 context clusters produced
    spurious FAILs in rehearsal; Theil-Sen is the right estimator here."""
    slopes = [(y2 - y1) / (x2 - x1)
              for i, (x1, y1) in enumerate(zip(xs, ys, strict=True))
              for x2, y2 in zip(xs[i + 1:], ys[i + 1:], strict=True)
              if x2 != x1]
    slopes.sort()
    return slopes[len(slopes) // 2] if slopes else 0.0


def _ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    var = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / var if var else 0.0
    return slope, my - slope * mx


def check_bandwidth(traces, fleet):
    """Batch-1 decode: TPOT ~= active weight bytes / effective bandwidth."""
    out = []
    for silicon in sorted({t.silicon for t in traces}):
        hw = fleet[silicon]
        obs = [t for t in traces if t.silicon == silicon and t.batch == 1]
        if not obs:
            out.append((silicon, None, "no batch-1 traces"))
            continue
        errs = []
        for t in obs:
            sig = t.signature()
            n_dev, tp = replica_layout(sig, hw)
            m = sig.model
            step_bytes = (m.active_params_b * 1e9 * m.bytes_per_param
                          + sig.avg_context * m.kv_bytes_per_token)
            pred_ms = step_bytes / (n_dev * hw.hbm_bw_tbs * 1e12 * hw.bw_eff * tp) * 1e3
            errs.append(abs(pred_ms - t.measured_tpot_ms) / t.measured_tpot_ms)
        e = statistics.median(errs)
        out.append((silicon, e, "PASS" if e <= TOL else "FAIL"))
    return out


def check_kv_slope(traces, fleet):
    """TPOT vs context at fixed batch: linear, slope = b*kv_bytes/BW_eff."""
    out = []
    groups = defaultdict(list)
    for t in traces:
        # Group by device layout too: linearity in context only holds while
        # n_dev is constant (KV growth can force a TP change mid-sweep, and
        # the slope is discontinuous across that boundary).
        hw = fleet[t.silicon]
        n_dev, _ = replica_layout(t.signature(), hw)
        groups[(t.silicon, t.model_name, t.batch, n_dev)].append(t)
    for (silicon, model, batch, n_dev), ts in sorted(groups.items()):
        ctxs = {t.signature().avg_context for t in ts}
        if len(ctxs) < 4:
            continue  # <4 distinct contexts: slope unidentifiable, silent skip
        hw = fleet[silicon]
        sig0 = ts[0].signature()
        _, tp = replica_layout(sig0, hw)
        m = sig0.model
        xs = [t.signature().avg_context for t in ts]
        ys = [t.measured_tpot_ms for t in ts]
        slope_meas = _theil_sen(xs, ys)
        # Power gate on the slope itself: bootstrap the estimator and SKIP
        # when the CI is too wide to support any verdict. Adapts to actual
        # measurement noise (unlike fixed context-count gates, which miss
        # clustered-context cells with formally many "distinct" values).
        rng = random.Random(0)
        boots = []
        idx = list(range(len(xs)))
        for _ in range(100):
            pick = [idx[rng.randrange(len(idx))] for _ in idx]
            boots.append(_theil_sen([xs[i] for i in pick], [ys[i] for i in pick]))
        boots.sort()
        lo, hi = boots[2], boots[97]
        if slope_meas <= 0 or (hi - lo) / (2 * abs(slope_meas)) > 0.5:
            out.append((f"{silicon}/{model}/b{batch}/tp{n_dev}", None,
                        "SKIP (slope CI too wide: underpowered)"))
            continue
        slope_pred = batch * m.kv_bytes_per_token / (
            n_dev * hw.hbm_bw_tbs * 1e12 * hw.bw_eff * tp) * 1e3
        if slope_pred <= 0:
            continue
        # Only meaningful in the memory-bound regime; skip compute-bound cells.
        e_probe = estimate(sig0, hw, hw.base_price_hr)
        if e_probe.bound != "memory":
            continue
        # Identifiability gate: the KV delta across the context range must be
        # a meaningful fraction of step bytes, or the slope is underpowered
        # at realistic measurement noise and any verdict is meaningless.
        lo, hi = min(xs), max(xs)
        kv_delta = batch * (hi - lo) * m.kv_bytes_per_token
        step_ref = (m.active_params_b * 1e9 * m.bytes_per_param
                    + batch * lo * m.kv_bytes_per_token)
        if kv_delta < 0.15 * step_ref:
            out.append((f"{silicon}/{model}/b{batch}/tp{n_dev}", None,
                        "SKIP (KV delta <15% of step bytes: underpowered)"))
            continue
        err = abs(slope_meas - slope_pred) / slope_pred
        out.append((f"{silicon}/{model}/b{batch}/tp{n_dev}", err,
                    "PASS" if err <= 2 * TOL else "FAIL"))  # slopes are noisier
    return out


def check_prefill(traces, fleet):
    """Per-point implied mfu from full prefill FLOPs (linear + L^2 attention).

    Per-point inversion instead of an OLS slope: TTFT is no longer linear in
    prompt length once the quadratic attention term is modeled, and per-point
    medians are robust to that curvature by construction. Additionally checks
    that implied mfu does not trend with prompt length (a trend = the L^2
    accounting is wrong on this stack, e.g. a flash-attention variant with
    different effective cost).
    """
    out = []
    groups = defaultdict(list)
    for t in traces:
        groups[(t.silicon, t.model_name)].append(t)
    for (silicon, model), ts in sorted(groups.items()):
        if len({t.avg_prompt_tokens for t in ts}) < 3:
            continue
        hw = fleet[silicon]
        pts = []
        for t in ts:
            sig = t.signature()
            n_dev, tp = replica_layout(sig, hw)
            mfu = sig.prefill_flops_per_req / (
                (t.measured_ttft_ms / 1e3) * n_dev * hw.peak_tflops * 1e12 * tp)
            pts.append((t.avg_prompt_tokens, mfu))
        med = statistics.median(m for _, m in pts)
        slope, _ = _ols([p for p, _ in pts], [m for _, m in pts])
        rel_trend = slope * (max(p for p, _ in pts) - min(p for p, _ in pts)) / med
        if not (0.0 < med <= 1.0):
            verdict = "FAIL (impossible mfu -> harness bug)"
        elif abs(rel_trend) > 2 * TOL:
            verdict = f"FAIL (mfu trends {rel_trend:+.0%} with L: attention accounting wrong)"
        else:
            verdict = "PASS"
        out.append((f"{silicon}/{model}", med, verdict))
    return out


def check_residuals(traces, fleet):
    """After calibration, grouped mean relative error should hug zero."""
    calibrated, report = calibrate(fleet, traces)
    rows = []
    by_batch, by_ctx = defaultdict(list), defaultdict(list)
    for t in traces:
        hw = calibrated[t.silicon]
        e = estimate(t.signature(), hw, hw.base_price_hr)
        if not e.feasible:
            continue
        r = (e.tpot_ms - t.measured_tpot_ms) / t.measured_tpot_ms
        by_batch[t.batch].append(r)
        c = t.signature().avg_context
        by_ctx["S" if c < 1024 else "M" if c <= 4096 else "L"].append(r)
    for k in sorted(by_batch):
        rows.append((f"batch={k}", statistics.mean(by_batch[k])))
    for k in ("S", "M", "L"):
        if k in by_ctx:
            rows.append((f"ctx={k}", statistics.mean(by_ctx[k])))
    return report, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+")
    p.add_argument("--bw-ceiling-gbps", type=float, help="microbenched bandwidth")
    p.add_argument("--flops-ceiling-tflops", type=float, help="microbenched GEMM")
    args = p.parse_args()

    traces = [t for f in args.files for t in load_jsonl(f)]
    print(f"loaded {len(traces)} traces\n")
    failed = False

    # Term checks test MODEL FORM (linearity, term structure), so they run
    # against the CALIBRATED fleet: prior-parameter offsets are calibration's
    # job to fix and would otherwise contaminate every slope verdict.
    fitted_fleet, _ = calibrate(FLEET, traces)

    print("== 1. bandwidth term (batch-1 decode) ==")
    for name, err, verdict in check_bandwidth(traces, fitted_fleet):
        print(f"  {name:<12} median err {err:.1%}  {verdict}" if err is not None
              else f"  {name:<12} {verdict}")
        failed |= verdict == "FAIL"

    print("== 2. KV term (TPOT-vs-context slope, memory-bound cells) ==")
    for name, err, verdict in check_kv_slope(traces, fitted_fleet):
        detail = f"slope err {err:.1%}  " if err is not None else ""
        print(f"  {name:<28} {detail}{verdict}")
        failed |= verdict == "FAIL"

    print("== 3. prefill term (TTFT slope -> implied mfu) ==")
    for name, mfu, verdict in check_prefill(traces, fitted_fleet):
        print(f"  {name:<28} implied mfu {mfu:.3f}  {verdict}")
        failed |= verdict.startswith("FAIL")

    print("== 4. calibration + residual structure ==")
    report, rows = check_residuals(traces, FLEET)
    print(f"  holdout MAPE: prior {report.mape_prior:.1%} -> "
          f"calibrated {report.mape_calibrated:.1%}")
    for label, r in rows:
        flag = "  <- structure" if abs(r) > 0.05 else ""
        print(f"  {label:<10} mean residual {r:+.1%}{flag}")

    if args.bw_ceiling_gbps or args.flops_ceiling_tflops:
        print("== 5. ceiling check (fitted <= microbenched <= peak) ==")
        for s, (_mfu, bw) in report.fitted.items():
            hw = FLEET[s]
            if args.bw_ceiling_gbps:
                fitted_bw = bw * hw.hbm_bw_tbs * 1000
                ok = fitted_bw <= args.bw_ceiling_gbps * 1.02
                print(f"  {s} fitted BW {fitted_bw:.0f} GB/s vs ceiling "
                      f"{args.bw_ceiling_gbps:.0f}: {'PASS' if ok else 'FAIL (harness bug)'}")
                failed |= not ok

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
