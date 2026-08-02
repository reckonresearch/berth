"""Fit the fixed prefill floor from measured batch-1 cells.

WHY this is a separate step rather than a shipped constant. TTFT decomposes as

    TTFT = fixed_floor + prompt_compute / achievable_flops

and the floor (scheduler admission, detokenizer setup, sampler construction,
first replay of a compiled CUDA graph) is prompt-length independent and NOT
predictable from a spec sheet. It was 74.6 ms on an L40S and 54.6 ms on an
H100 PCIe under vLLM 0.6.3: same vendor, same model, same harness, 37 percent
apart. It is a property of the (accelerator, driver, server, config) tuple,
so silicon profiles ship 0.0 and you fit your own.

Reusing someone else's floor is the same error as reusing someone else's
measurement, and it is the error that made the P0 prefill check report an
"attention accounting" failure that was never a model failure at all.

Method: Theil-Sen slope of TTFT against prompt tokens over batch-1 cells only
(batched cells carry the serial-admission term and would bias the intercept
upward), then the median residual as the intercept. Robust to the handful of
points a real sweep produces.

Usage:
    python -m bench.fit_overhead traces.jsonl [more.jsonl ...] [--silicon KEY]

Feed the result straight back in:
    python -m bench.validate traces.jsonl --prefill-overhead-ms 54.6
"""

import argparse
import statistics
import sys
from collections import defaultdict

from bench.sounding import load_jsonl, provenance_of
from bench.validate import _theil_sen

MIN_CELLS = 3          # a slope through two points is not a fit
MIN_SPREAD = 2.0       # max/min prompt length; a flat sweep cannot separate
                       # the floor from the slope, and would report noise


def fit_floor(traces):
    """Per-silicon fitted floor in ms, plus the diagnostics to judge it.

    Returns {silicon: dict} where a dict with 'error' means we refused to fit
    rather than returning a number we cannot defend.
    """
    by_silicon = defaultdict(list)
    for t in traces:
        if t.batch == 1:
            by_silicon[t.silicon].append(t)

    out = {}
    for silicon, cells in sorted(by_silicon.items()):
        # Average repetitions of the same prompt length before fitting, so a
        # length measured twice does not outvote one measured once.
        by_len = defaultdict(list)
        for c in cells:
            by_len[c.avg_prompt_tokens].append(c.measured_ttft_ms)
        xs = sorted(by_len)
        ys = [statistics.median(by_len[x]) for x in xs]

        if len(xs) < MIN_CELLS:
            out[silicon] = {"error": f"only {len(xs)} distinct prompt lengths at "
                                     f"batch 1; need {MIN_CELLS}"}
            continue
        spread = xs[-1] / xs[0] if xs[0] else float("inf")
        if spread < MIN_SPREAD:
            out[silicon] = {"error": f"prompt-length spread {spread:.1f}x is too "
                                     f"flat to separate floor from slope; need "
                                     f"{MIN_SPREAD}x"}
            continue

        slope = _theil_sen(xs, ys)                       # ms per prompt token
        residuals = [y - slope * x for x, y in zip(xs, ys, strict=True)]
        floor = statistics.median(residuals)
        out[silicon] = {
            "floor_ms": floor,
            "slope_ms_per_token": slope,
            "n_lengths": len(xs),
            "n_cells": len(cells),
            "spread": spread,
            "residual_spread_ms": max(residuals) - min(residuals),
            "negative": floor < 0,
        }
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m bench.fit_overhead",
        description="Fit the fixed prefill floor from batch-1 cells.")
    p.add_argument("files", nargs="+")
    p.add_argument("--silicon", help="restrict to one silicon key")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args(argv)

    traces = [t for f in args.files for t in load_jsonl(f)]
    if args.silicon:
        traces = [t for t in traces if t.silicon == args.silicon]
    if not traces:
        print("no traces matched", file=sys.stderr)
        return 2

    kind = provenance_of(traces)
    fits = fit_floor(traces)

    if args.json:
        import json
        print(json.dumps({"provenance": kind, "fits": fits}, indent=2))
        return 0 if any("floor_ms" in v for v in fits.values()) else 1

    print(f"prefill floor, fitted from batch-1 cells   [provenance: {kind}]")
    if kind == "mock":
        print("  WARNING: mock traces. This floor is whatever the generator "
              "assumed, not a hardware property.")
    if not fits:
        print("  no batch-1 cells found; the floor cannot be fitted from "
              "batched traces (they carry the serial-admission term)")
        return 1

    ok = False
    for silicon, f in fits.items():
        if "error" in f:
            print(f"  {silicon:<12} SKIP  {f['error']}")
            continue
        ok = True
        print(f"  {silicon:<12} floor {f['floor_ms']:7.1f} ms   "
              f"slope {f['slope_ms_per_token']*1000:6.3f} ms/kilotoken   "
              f"{f['n_lengths']} lengths / {f['n_cells']} cells   "
              f"spread {f['spread']:.1f}x")
        if f["negative"]:
            print("               ^ NEGATIVE floor. The slope is over-attributing "
                  "compute; do not use this value. Check the prompt lengths are "
                  "the served counts, not the requested ones.")
        if f["residual_spread_ms"] > abs(f["floor_ms"]):
            print("               ^ residual spread exceeds the floor itself; "
                  "the fit is not tight. Widen the prompt-length range.")

    print("\nApply it:  python -m bench.validate <files> --prefill-overhead-ms "
          "<floor for the silicon you measured>")
    print("The floor belongs to one (accelerator, driver, server, config). "
          "Do not carry it to another.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
