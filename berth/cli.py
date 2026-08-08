"""berth command-line interface.

A thin argparse front-end over the public API (estimate, FLEET, MODELS), so a
first number takes one line and no Python. Every command prints provenance:
MEASURED silicon versus spec-sheet prior. Installed as the `berth` entry point.
"""

import argparse
import json
import sys

from berth.estimate import estimate
from berth.silicon import FLEET
from berth.workload import MODELS, WorkloadSpec, profile

# Silicon validated against real hardware traces (P0). Everything else is a
# spec-sheet prior, and the CLI says so on every line it prints.
MEASURED = {"l40s", "h100-pcie"}


# Column layout is shared between the header and the rows so they stay aligned,
# and so the output is addressable by field number in awk. Units live in the
# header rather than repeated on every row: a reader learns them once, and a
# script does not have to strip "ms" off every value.
_HEADER = (f"{'silicon':<12} {'$/Mtok':>9}  {'TTFT_ms':>9}  {'TPOT_ms':>9}  "
           f"{'tok/s':>9}  provenance")


def _fmt_estimate(key, e, price_hr):
    tag = "MEASURED" if key in MEASURED else "prior"
    return (f"{key:<12} {e.cost_per_mtok:9.3f}  {e.ttft_ms:9.1f}  "
            f"{e.tpot_ms:9.2f}  {e.tokens_per_s:9.1f}  {tag}")


def cmd_estimate(args):
    if args.model not in MODELS:
        sys.exit(f"unknown model '{args.model}'. known: {', '.join(MODELS)}")
    sig = profile(WorkloadSpec(
        model=MODELS[args.model],
        avg_prompt_tokens=args.prompt,
        avg_output_tokens=args.output,
        target_batch=args.batch,
    ))
    keys = [args.silicon] if args.silicon else list(FLEET)
    for k in keys:
        if k not in FLEET:
            sys.exit(f"unknown silicon '{k}'. known: {', '.join(FLEET)}")
    price = {k: args.price_hr for k in keys} if args.price_hr else {
        k: FLEET[k].base_price_hr for k in keys}
    rows = [(k, estimate(sig, FLEET[k], price[k])) for k in keys]
    if args.json:
        print(json.dumps([{
            "silicon": k, "measured": k in MEASURED,
            "cost_per_mtok": e.cost_per_mtok, "ttft_ms": e.ttft_ms,
            "tpot_ms": e.tpot_ms, "tokens_per_s": e.tokens_per_s,
            "feasible": e.feasible, "n_devices": e.n_devices,
        } for k, e in rows], indent=2))
        return
    print(f"# {args.model}  batch={args.batch}  prompt={args.prompt}  output={args.output}")
    print(_HEADER)
    for k, e in sorted(rows, key=lambda r: r[1].cost_per_mtok):
        print(_fmt_estimate(k, e, price[k]))
    print("# provenance: MEASURED = validated against hardware traces; "
          "prior = spec sheet, unverified")


def cmd_premium(args):
    if args.model not in MODELS:
        sys.exit(f"unknown model '{args.model}'. known: {', '.join(MODELS)}")
    sig = profile(WorkloadSpec(
        model=MODELS[args.model],
        avg_prompt_tokens=args.prompt,
        avg_output_tokens=args.output,
        target_batch=args.batch,
    ))
    keys = args.silicon or list(FLEET)
    for k in keys:
        if k not in FLEET:
            sys.exit(f"unknown silicon '{k}'. known: {', '.join(FLEET)}")
    price = {k: FLEET[k].base_price_hr for k in keys}
    for kv in (args.prices or []):
        kk, _, vv = kv.partition("=")
        if kk not in FLEET:
            sys.exit(f"unknown silicon '{kk}' in --prices")
        price[kk] = float(vv)
    rows = [(k, estimate(sig, FLEET[k], price[k])) for k in keys]
    feasible = [(k, e) for k, e in rows if e.feasible]
    if not feasible:
        sys.exit("no feasible placement in the requested set")
    cheapest_k, cheapest_e = min(feasible, key=lambda r: r[1].cost_per_mtok)
    print(f"# {args.model}  batch={args.batch}  cheapest feasible: {cheapest_k}")
    print("# numbers are PREDICTED (berth roofline). For measured premia see the "
          "published traces; pass real prices with --prices to match your deployment.")
    for k, e in sorted(feasible, key=lambda r: r[1].cost_per_mtok):
        prem = e.cost_per_mtok / cheapest_e.cost_per_mtok
        tag = "MEASURED" if k in MEASURED else "prior"
        star = "  <- cheapest" if k == cheapest_k else ""
        print(f"{k:<12} {e.cost_per_mtok:8.3f} $/Mtok   premium {prem:5.2f}x   [{tag}]{star}")


def cmd_list(args):
    print("# silicon (MEASURED = validated on real hardware; prior = spec sheet)")
    for k, hw in FLEET.items():
        tag = "MEASURED" if k in MEASURED else "prior"
        print(f"  {k:<12} peak {hw.peak_tflops:>5.0f} TFLOPS   bw {hw.hbm_bw_tbs:>4.2f} TB/s   "
              f"{hw.mem_gb:>3}GB   ${hw.base_price_hr:>5.2f}/hr   [{tag}]")
    print("# models")
    for k in MODELS:
        print(f"  {k}")


def cmd_holdout(args):
    """Check that the assignment distributes over a customer's identifiers.

    Run before a period opens. A hash that does not spread evenly over their
    identifier format is a defect in the instrument rather than a result, and
    finding it here costs nothing while finding it in settlement voids a
    month.
    """
    from berth.holdout import assign, realised_fraction, scale_floor

    if args.ids:
        with open(args.ids) as f:
            ids = [ln.strip() for ln in f if ln.strip()]
        if not ids:
            raise SystemExit(f"{args.ids} contains no identifiers")
    else:
        import uuid
        ids = [str(uuid.uuid4()) for _ in range(args.n)]
        print(f"no --ids given, checking against {args.n:,} synthetic UUIDs. "
              f"Use a sample of your real identifiers instead: sequential and "
              f"tenant-prefixed formats behave differently.")

    got = realised_fraction(ids, args.seed, args.fraction)
    drift = abs(got - args.fraction)
    print(f"\n  identifiers          {len(ids):,}")
    print(f"  declared fraction    {args.fraction:.4f}")
    print(f"  realised fraction    {got:.4f}")
    print(f"  drift                {drift:.4f}")

    if drift > 0.01:
        print("\n  OUT OF TOLERANCE. The realised split is more than one "
              "percentage point from the declared one, which means the hash "
              "is not spreading evenly over this identifier format. Do not "
              "open a period on it.")
        return 1
    print("\n  within tolerance.")

    if args.spend:
        s_star = scale_floor(args.share)
        inst = ("assurance_light" if args.spend < 250_000
                else "assurance" if args.spend < s_star else "savings_share")
        print(f"\n  class spend          ${args.spend:,.0f}")
        print(f"  scale floor          ${s_star:,.0f}")
        print(f"  instrument           {inst}")
        if inst != "savings_share":
            print("  Below the floor the holdout costs more than the "
                  "arrangement returns, so a flat fee is the right "
                  "instrument and no holdout is needed.")

    print("\n  sample assignments, reproducible from the seed:")
    for r in ids[:5]:
        print(f"    {r[:36]:<38}{assign(r, args.seed, args.fraction)}")
    return 0


def build_parser():
    p = argparse.ArgumentParser(
        prog="berth",
        description="Predict inference cost and latency before you rent the GPU.")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("estimate", help="estimate one workload on one or all silicon")
    e.add_argument("--model", required=True, help="model key, e.g. llama3-8b")
    e.add_argument("--silicon", help="silicon key; omit to score the whole fleet")
    e.add_argument("--batch", type=int, default=1)
    e.add_argument("--prompt", type=int, default=2048, help="avg prompt tokens")
    e.add_argument("--output", type=int, default=256, help="avg output tokens")
    e.add_argument("--price-hr", type=float, help="override $/hr (default: fleet list price)")
    e.add_argument("--json", action="store_true", help="machine-readable output")
    e.set_defaults(func=cmd_estimate)

    pr = sub.add_parser("premium", help="rank placements by cost, show the premium")
    pr.add_argument("--model", required=True)
    pr.add_argument("--silicon", nargs="*", help="silicon keys to compare; omit for whole fleet")
    pr.add_argument("--batch", type=int, default=32)
    pr.add_argument("--prompt", type=int, default=2048)
    pr.add_argument("--output", type=int, default=256)
    pr.add_argument("--prices", nargs="*", metavar="KEY=USD",
                    help="per-silicon price overrides, e.g. l40s=0.99 h100-pcie=3.35")
    pr.set_defaults(func=cmd_premium)

    ho = sub.add_parser("holdout",
                        help="check a holdout assignment before opening a period")
    ho.add_argument("--seed", required=True,
                    help="the seed from the declaration, committed in advance")
    ho.add_argument("--fraction", type=float, default=0.05,
                    help="declared holdout fraction")
    ho.add_argument("--ids", help="file of real request identifiers, one per line")
    ho.add_argument("--n", type=int, default=50_000,
                    help="synthetic identifiers to use when --ids is absent")
    ho.add_argument("--spend", type=float,
                    help="annual class spend, to pick the instrument")
    ho.add_argument("--share", type=float, default=0.20)
    ho.set_defaults(func=cmd_holdout)

    ls = sub.add_parser("list", help="list known silicon and models with provenance")
    ls.set_defaults(func=cmd_list)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
