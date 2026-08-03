"""Audit a trace file for contamination before trusting anything fitted from it.

WHY this exists as a separate tool. The four instrument defects found so far
all shared a shape: the run completed, the numbers looked plausible, and a
downstream check reported a model failure. Nothing in the pipeline asked
whether the measurements were measurements.

These checks are physics, not statistics. Each one names a quantity that
cannot be true and says what produces it. A trace file that passes all of
them may still be wrong, but it is not wrong in any of the ways that have
already cost a day.

    python -m bench.audit_traces traces.jsonl --peak-tflops 312 --bw-tbs 2.0

Exit 0 clean, 1 contaminated, 2 unusable input.
"""

import argparse
import json
import statistics as st
import sys
from collections import defaultdict

from bench.sounding import load_jsonl


def _median_by(traces, keyfn, valfn):
    d = defaultdict(list)
    for t in traces:
        d[keyfn(t)].append(valfn(t))
    return {k: st.median(v) for k, v in d.items()}


def check_prefill_possible(traces, peak_tflops, active_params_b):
    """TTFT implies a prefill rate. That rate cannot exceed the card's peak.

    Detects prefix caching, which removes the prefill from TTFT entirely while
    leaving a plausible-looking number behind. On an A100 running an MoE this
    produced apparent throughput of 18.8x peak.
    """
    problems = []
    worst = 0.0
    for t in traces:
        toks = t.batch * t.avg_prompt_tokens
        secs = t.measured_ttft_ms / 1000.0
        if secs <= 0:
            continue
        tflops = (2 * active_params_b * 1e9 * toks) / secs / 1e12
        worst = max(worst, tflops / peak_tflops)
    if worst > 1.0:
        problems.append(
            f"implied prefill throughput reaches {worst:.1f}x the card's peak "
            f"FLOPS, which is impossible. TTFT is not measuring prefill. The "
            f"usual cause is identical prompts plus automatic prefix caching: "
            f"the first request prefills and the rest are served from cache.")
    return problems, worst


def check_ttft_scales_with_tokens(traces):
    """Prefill work is linear in tokens. TTFT should be too, above the floor.

    A near-flat TTFT across a large token range is the prefix-cache signature
    and is visible even when the absolute numbers look reasonable.
    """
    problems = []
    by_batch = defaultdict(lambda: defaultdict(list))
    for t in traces:
        by_batch[t.batch][t.avg_prompt_tokens].append(t.measured_ttft_ms)
    for b, lens in sorted(by_batch.items()):
        if len(lens) < 2:
            continue
        ps = sorted(lens)
        tok_ratio = ps[-1] / ps[0]
        t_ratio = st.median(lens[ps[-1]]) / st.median(lens[ps[0]])
        # A fixed floor flattens this legitimately, so the bar is low: a 15x
        # token increase costing under 1.5x time cannot be explained by a
        # floor unless the floor is nearly all of TTFT, which is itself worth
        # knowing.
        if tok_ratio >= 5 and t_ratio < 1.5:
            problems.append(
                f"batch {b}: {tok_ratio:.0f}x the prompt tokens costs only "
                f"{t_ratio:.2f}x the time. Prefill is not being measured, or "
                f"the fixed floor dominates TTFT at every length in this sweep.")
    return problems


def check_bw_eff_is_constant(traces, bw_tbs, active_params_b, kv_bytes_per_token):
    """Effective bandwidth is a device property. It cannot drift with context.

    Drift with context at batch > 1, while batch 1 stays flat, is the
    shared-KV-block signature: identical prompts let the server read the
    prefix once per step instead of once per request, so the KV term appears
    overcounted and the drift grows with batch.
    """
    problems = []
    rows = {}
    by = defaultdict(lambda: defaultdict(list))
    for t in traces:
        by[t.batch][t.avg_prompt_tokens].append(t)
    for b, lens in sorted(by.items()):
        effs = []
        for p in sorted(lens):
            tp = st.median(x.measured_tpot_ms for x in lens[p])
            ctx = p + lens[p][0].avg_output_tokens // 2
            floor_ms = (active_params_b * 1e9 * 2 + b * ctx * kv_bytes_per_token) \
                / (bw_tbs * 1e12) * 1000
            if tp > 0:
                effs.append(floor_ms / tp)
        if len(effs) >= 2:
            spread = max(effs) / min(effs)
            rows[b] = (effs, spread)
            if spread > 1.20:
                problems.append(
                    f"batch {b}: implied bw_eff drifts {spread:.2f}x across "
                    f"context ({min(effs):.2f} to {max(effs):.2f}). Effective "
                    f"bandwidth is a device constant; drift means the KV term "
                    f"is being fitted against bytes that were not moved.")
        if effs and max(effs) > 1.0:
            problems.append(
                f"batch {b}: implied bw_eff reaches {max(effs):.2f}, above 1.0. "
                f"The card cannot exceed its own bandwidth; the byte count is "
                f"too high or the timing is too short.")
    return problems, rows


def check_cell_coverage(traces):
    """Every cell needs enough repetitions to separate signal from jitter, and
    a holdout that splits repetitions rather than cells proves nothing."""
    problems = []
    cells = defaultdict(int)
    for t in traces:
        cells[(t.silicon, t.model_name, t.batch, t.avg_prompt_tokens,
               t.avg_output_tokens)] += 1
    thin = [c for c, n in cells.items() if n < 2]
    if thin:
        problems.append(
            f"{len(thin)} of {len(cells)} cells have a single observation. "
            f"A cell measured once cannot be distinguished from jitter, and a "
            f"holdout split over such a file separates repetitions rather than "
            f"cells.")
    return problems, len(cells)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m bench.audit_traces",
        description="Check a trace file for measurement contamination.")
    p.add_argument("files", nargs="+")
    p.add_argument("--peak-tflops", type=float, required=True,
                   help="dense bf16 peak of the card, e.g. 312 for A100-80G")
    p.add_argument("--bw-tbs", type=float, required=True,
                   help="memory bandwidth in TB/s, e.g. 2.0 for A100-80G")
    p.add_argument("--active-params-b", type=float, required=True,
                   help="active parameters in billions (= total, if dense)")
    p.add_argument("--kv-bytes-per-token", type=float, required=True,
                   help="2 * n_layers * n_kv_heads * head_dim * bytes_per_kv")
    args = p.parse_args(argv)

    traces = [t for f in args.files for t in load_jsonl(f)]
    if not traces:
        print("no traces", file=sys.stderr)
        return 2

    print(f"auditing {len(traces)} traces\n")
    all_problems = []

    probs, worst = check_prefill_possible(traces, args.peak_tflops,
                                          args.active_params_b)
    print(f"  prefill within peak       worst {worst:.2f}x peak   "
          f"{'FAIL' if probs else 'ok'}")
    all_problems += probs

    probs = check_ttft_scales_with_tokens(traces)
    print(f"  TTFT scales with tokens   {'FAIL' if probs else 'ok'}")
    all_problems += probs

    probs, rows = check_bw_eff_is_constant(
        traces, args.bw_tbs, args.active_params_b, args.kv_bytes_per_token)
    print(f"  bw_eff constant           {'FAIL' if probs else 'ok'}")
    for b, (effs, spread) in sorted(rows.items()):
        print(f"      b={b:<3} " + " ".join(f"{e:.2f}" for e in effs)
              + f"   spread {spread:.2f}x")
    all_problems += probs

    probs, ncells = check_cell_coverage(traces)
    print(f"  cell coverage             {ncells} cells   "
          f"{'FAIL' if probs else 'ok'}")
    all_problems += probs

    if all_problems:
        print(f"\nCONTAMINATED: {len(all_problems)} finding"
              f"{'s' if len(all_problems) != 1 else ''}\n")
        for x in all_problems:
            print(f"  x  {x}\n")
        print("Do not fit anything to this file until these are resolved. A "
              "calibration over contaminated traces produces a model that "
              "predicts the contamination.")
        return 1

    print("\nCLEAN on every check above. That is not a guarantee of "
          "correctness, only that this file is not wrong in any of the ways "
          "that have been wrong before.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
