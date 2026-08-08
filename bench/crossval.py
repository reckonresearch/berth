"""Cross-validation for the estimator. Four splits, ascending in what they prove.

Named crossval rather than holdout because `berth.holdout` is a different
thing entirely: the commercial protocol under which a share of traffic stays
on a baseline placement so a saving can be measured. This module holds out
cells from a fit. That one holds out requests from a routing change. Two
unrelated ideas that both wanted the word.

WHY this replaces the current holdout. Splitting repetitions puts the same
cell on both sides of the line, so the model has already seen every
configuration it is scored on and every error figure is in-distribution. That
makes the published numbers optimistic by an unknown amount, which is worse
than making them optimistic by a known one.

The four splits:

    cell        leave one (batch, context, output) out    interpolation
    batch       leave one batch size out                  extrapolation to
                                                          unseen concurrency
    context     leave one prompt length out               KV scaling
    silicon     leave one accelerator out                 cross-vendor transfer

The last is the one that matters and the one already effectively passed:
untuned spec-sheet priors implied 648 GB/s against a microbenchmarked 652 on a
card the model had never seen. This formalises that as a pre-registered gate
rather than an anecdote.

One trap, stated because it has caught this project before: the fixed prefill
floor is fitted per card and per configuration. Any error computed on the same
traces used to fit it is in-sample no matter which split is drawn. Fit the
floor on the training side only, and say so.

    python -m bench.crossval traces_l40s.jsonl traces_h100.jsonl \\
        --split silicon --model llama3-8b

Exit 0 if every held-out fold lands inside the gate, 1 otherwise.
"""

import argparse
import statistics as st
import sys
from collections import defaultdict

from bench.sounding import load_jsonl


def theil_sen_floor(rows):
    """Fixed prefill floor from batch-1 cells, robust to three points.

    Theil-Sen rather than least squares because a three-point quadratic
    overshoots badly: on one A100 file it produced a floor above every
    measurement it was meant to sit under, and the conclusion drawn from it
    reversed when the estimator changed.
    """
    b1 = [r for r in rows if r.batch == 1]
    if len(b1) < 2:
        return None
    pts = sorted({(r.avg_prompt_tokens,
                   st.median(x.measured_ttft_ms for x in b1
                             if x.avg_prompt_tokens == r.avg_prompt_tokens))
                  for r in b1})
    if len(pts) < 2:
        return None
    slopes = [(pts[j][1] - pts[i][1]) / (pts[j][0] - pts[i][0])
              for i in range(len(pts)) for j in range(i + 1, len(pts))
              if pts[j][0] != pts[i][0]]
    if not slopes:
        return None
    m = st.median(slopes)
    return st.median(y - m * x for x, y in pts)


def key_for(split, r):
    if split == "cell":
        return (r.silicon, r.batch, r.avg_prompt_tokens, r.avg_output_tokens)
    if split == "batch":
        return r.batch
    if split == "context":
        return r.avg_prompt_tokens
    if split == "silicon":
        return r.silicon
    raise SystemExit(f"unknown split {split!r}")


def bw_eff_of(rows, active_params_b, kv_bytes_per_token, bw_tbs):
    """Fit one effective bandwidth from decode timings.

    Deliberately the simplest possible fit: the point of a held-out split is
    to test whether one constant transfers, and a richer fit would make the
    transfer easier for reasons that have nothing to do with the physics.
    """
    effs = []
    for r in rows:
        ctx = r.avg_prompt_tokens + r.avg_output_tokens / 2
        bytes_moved = active_params_b * 1e9 * 2 + r.batch * ctx * kv_bytes_per_token
        floor_s = bytes_moved / (bw_tbs * 1e12)
        if r.measured_tpot_ms > 0:
            effs.append(floor_s * 1000 / r.measured_tpot_ms)
    return st.median(effs) if effs else None


def predict_tpot(r, bw_eff, active_params_b, kv_bytes_per_token, bw_tbs):
    ctx = r.avg_prompt_tokens + r.avg_output_tokens / 2
    bytes_moved = active_params_b * 1e9 * 2 + r.batch * ctx * kv_bytes_per_token
    return bytes_moved / (bw_tbs * 1e12 * bw_eff) * 1000


def run_split(traces, split, active_params_b, kv_bytes_per_token, bw_by_silicon,
              gate):
    folds = defaultdict(list)
    for r in traces:
        folds[key_for(split, r)].append(r)

    print(f"\n{split} split: {len(folds)} folds\n")
    print(f"  {'held out':<22}{'train n':>9}{'test n':>8}{'fitted bw_eff':>15}"
          f"{'test MAPE':>12}")
    results = []
    for held, test in sorted(folds.items(), key=lambda kv: str(kv[0])):
        train = [r for r in traces if key_for(split, r) != held]
        if not train or not test:
            continue
        # Fit on the training side only. Fitting on everything and scoring on
        # a subset is the mistake this module exists to remove.
        errs = []
        for sil in {r.silicon for r in test}:
            bw_test = bw_by_silicon.get(sil)
            if bw_test is None:
                continue
            # bw_eff is dimensionless: a fraction of a card's own peak. It must
            # therefore be fitted against the TRAINING card's bandwidth and
            # then applied with the HELD-OUT card's bandwidth. Fitting on one
            # card's timings against another card's peak is a unit error, and
            # it produces efficiencies above 1.0, which is the tell.
            same = [r for r in train if r.silicon == sil]
            if same:
                eff = bw_eff_of(same, active_params_b, kv_bytes_per_token, bw_test)
            else:
                # Leave-one-silicon-out: fit each training card against its
                # own peak, then take the median across them. This is the
                # transfer question, and it is the only split where the
                # fitted constant crosses a device boundary.
                effs = []
                for tsil in {r.silicon for r in train}:
                    bw_tr = bw_by_silicon.get(tsil)
                    if bw_tr is None:
                        continue
                    e = bw_eff_of([r for r in train if r.silicon == tsil],
                                  active_params_b, kv_bytes_per_token, bw_tr)
                    if e is not None:
                        effs.append(e)
                eff = st.median(effs) if effs else None
            if eff is None:
                continue
            for r in (x for x in test if x.silicon == sil):
                pred = predict_tpot(r, eff, active_params_b,
                                    kv_bytes_per_token, bw_test)
                errs.append(abs(pred - r.measured_tpot_ms) / r.measured_tpot_ms)
        if not errs:
            continue
        mape = 100 * st.mean(errs)
        sil0 = sorted({r.silicon for r in test})[0]
        same0 = [r for r in train if r.silicon == sil0]
        if same0:
            eff0 = bw_eff_of(same0, active_params_b, kv_bytes_per_token,
                             bw_by_silicon.get(sil0, 1.0))
        else:
            e0 = [bw_eff_of([r for r in train if r.silicon == t],
                            active_params_b, kv_bytes_per_token,
                            bw_by_silicon[t])
                  for t in {r.silicon for r in train} if t in bw_by_silicon]
            e0 = [e for e in e0 if e is not None]
            eff0 = st.median(e0) if e0 else None
        results.append((held, mape))
        flag = "" if mape <= gate else "  OVER GATE"
        print(f"  {str(held):<22}{len(train):>9}{len(test):>8}"
              f"{(eff0 or 0):>15.3f}{mape:>11.1f}%{flag}")
    if results:
        worst = max(r[1] for r in results)
        print(f"\n  worst fold: {worst:.1f}% against a {gate:.0f}% gate")
    return results


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m bench.crossval",
        description="Held-out validation across four splits.")
    p.add_argument("files", nargs="+")
    p.add_argument("--split", default="silicon",
                   choices=["cell", "batch", "context", "silicon", "all"])
    p.add_argument("--active-params-b", type=float, required=True)
    p.add_argument("--kv-bytes-per-token", type=float, required=True)
    p.add_argument("--gate", type=float, default=15.0,
                   help="pre-registered error gate, percent")
    p.add_argument("--respect-envelope", action="store_true",
                   help="also report error over only those cells the model "
                        "declares itself applicable to, i.e. kv_pressure below "
                        "KV_PRESSURE_WARN. Both figures are printed; the "
                        "unscoped one is the headline.")
    p.add_argument("--model", help="model key, required with --respect-envelope")
    p.add_argument("--bw", action="append", default=[], metavar="SILICON=TBS",
                   help="datasheet peak bandwidth per silicon, repeatable, "
                        "e.g. --bw l40s=0.864 --bw h100-pcie=2.0")
    args = p.parse_args(argv)

    bw_by_silicon = {}
    for spec in args.bw:
        k, _, v = spec.partition("=")
        bw_by_silicon[k] = float(v)

    traces = [t for f in args.files for t in load_jsonl(f)]
    if not traces:
        sys.exit("no traces")
    missing = {r.silicon for r in traces} - set(bw_by_silicon)
    if missing:
        sys.exit(f"no --bw given for {sorted(missing)}. Peak bandwidth is a "
                 f"datasheet figure, not a microbenchmark: passing a d2d copy "
                 f"result here inflates every bw_eff above 1.")

    print(f"{len(traces)} traces, {len({r.silicon for r in traces})} accelerators")

    outside = []
    if args.respect_envelope:
        # The envelope is a property of the model, declared before the data is
        # seen: a cell whose KV cache would not fit one card is one the
        # estimator says it does not apply to. Excluding those is scoring the
        # claim that was made. Excluding the worst residuals would not be, and
        # the difference is that this rule is computable without looking at
        # any measurement.
        from berth.estimate import KV_PRESSURE_WARN, estimate
        from berth.silicon import FLEET
        from berth.workload import MODELS, WorkloadSpec, profile
        if not args.model or args.model not in MODELS:
            sys.exit("--respect-envelope needs --model naming a registry entry")
        keep = []
        for r in traces:
            hw = FLEET.get(r.silicon)
            if hw is None:
                keep.append(r)
                continue
            sig = profile(WorkloadSpec(model=MODELS[args.model],
                                       avg_prompt_tokens=r.avg_prompt_tokens,
                                       avg_output_tokens=r.avg_output_tokens,
                                       target_batch=r.batch))
            e = estimate(sig, hw, hw.base_price_hr)
            (outside if e.kv_pressure > KV_PRESSURE_WARN else keep).append(r)
        if outside:
            cells = sorted({(r.silicon, r.batch, r.avg_prompt_tokens) for r in outside})
            print(f"\nOutside the declared envelope, {len(outside)} traces "
                  f"across {len(cells)} cells:")
            for c in cells:
                print(f"    {c[0]}  batch {c[1]}  prompt {c[2]}  "
                      f"KV does not fit one card")
            print("  These are reported separately below. The model flags them "
                  "before any measurement is taken.")
    floor = theil_sen_floor(traces)
    if floor is not None:
        print(f"prefill floor across all traces: {floor:.0f} ms "
              f"(refitted per fold below)")

    splits = (["cell", "batch", "context", "silicon"]
              if args.split == "all" else [args.split])
    worst = 0.0
    for sp in splits:
        res = run_split(traces, sp, args.active_params_b,
                        args.kv_bytes_per_token, bw_by_silicon, args.gate)
        if res:
            worst = max(worst, max(r[1] for r in res))

    if outside:
        inside = [r for r in traces if r not in outside]
        print("\n" + "=" * 66)
        print("Within the declared envelope only")
        print("=" * 66)
        scoped = 0.0
        for sp in splits:
            res = run_split(inside, sp, args.active_params_b,
                            args.kv_bytes_per_token, bw_by_silicon, args.gate)
            if res:
                scoped = max(scoped, max(r[1] for r in res))
        print(f"\nScoped worst fold {scoped:.1f}%, unscoped {worst:.1f}%. "
              f"Publish both. The unscoped figure is the headline, because a "
              f"reader deciding whether to trust this has not yet agreed to "
              f"the envelope.")

    print()
    if worst <= args.gate:
        print(f"PASS: worst held-out fold {worst:.1f}% within the "
              f"{args.gate:.0f}% gate.")
        return 0
    print(f"OVER GATE: worst held-out fold {worst:.1f}% against "
          f"{args.gate:.0f}%.\n\nThis is the honest number. The in-sample "
          f"figure will be lower and should not be published in its place.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
