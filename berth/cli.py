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


# Silicon validated against real hardware traces. Everything else is a
# spec-sheet prior, and the CLI says so on every line it prints.
#
# Derived from the corpus rather than typed. This was frozen at the two P0
# cards and never updated as cells landed, so the CLI, the public Index, and
# every artifact downstream of them understated the corpus by three
# accelerators for weeks. A card whose first cross-vendor measurement is
# labelled spec-sheet arithmetic is the same error as an unmeasured card
# labelled measured, in the direction that costs credibility rather than
# accuracy.
#
# A card is MEASURED when at least one cell exists for it. It does not mean
# every model on it has been measured, and the band on a given placement says
# which.
def _measured_silicon():
    """Derived from the corpus rather than typed.

    This was frozen at the two P0 cards and never updated as cells landed, so
    the CLI, the public Index, and every artifact downstream understated the
    corpus by three accelerators for weeks.
    """
    from berth.place import MEASURED_CELLS
    return {silicon for silicon, _model in MEASURED_CELLS}


MEASURED = _measured_silicon()


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


def cmd_place(args):
    """Emit a placement decision record.

    Distinct from `estimate`, which answers what one workload costs on one
    accelerator. A decision names what was recommended, what it beat, by how
    much, and whether that margin survives the uncertainty on it. A contract
    is written against a decision; an estimate that moves when the corpus
    moves is correct behaviour, while a decision that moves silently is a
    dispute.
    """
    import json as _json

    from berth.place import decide, render
    rec = decide(workload_class=args.workload_class, model_key=args.model,
                 incumbent=args.incumbent, slo_bound_ms=args.slo_ms,
                 batch=args.batch, prompt_tokens=args.prompt,
                 output_tokens=args.output)
    if args.json:
        from dataclasses import asdict
        print(_json.dumps(asdict(rec), indent=2))
    else:
        print(render(rec))
    return 0 if rec.clears_band else 0


def cmd_pilot(args):
    """Run one pass of the agent against a declaration.

    Shadow by default. pilot changes production configuration, and the correct
    first posture is to read what it would have said for a while before
    letting it say anything.

    This read the declaration with json.load for a file that is YAML, so the
    command every customer would run to use pilot crashed on its first line.
    The declaration parser existed and the CLI never called it.
    """
    import json as _json
    from pathlib import Path

    from berth.agent import AgentState, run
    from berth.declaration import load_yaml
    from berth.place import decide
    from berth.watch import WatchState, build_detector

    path = Path(args.classes)
    if not path.exists():
        raise SystemExit(f"no declaration at {path}")
    decl = load_yaml(path.read_text(), repo=args.repo or "")

    ws = WatchState()
    ags = AgentState()
    if args.state and Path(args.state).exists():
        saved = _json.loads(Path(args.state).read_text())
        ws.model_versions = saved.get("model_versions", {})
        ws.stack_versions = saved.get("stack_versions", {})
        ws.prices = saved.get("prices", {})
        ws.corpus_cells = {tuple(c) for c in saved.get("corpus_cells", [])}
    elif args.state:
        print(f"no state at {args.state}. The first pass records what every "
              f"source looks like and fires nothing, because we do not yet "
              f"know whether anything changed.")

    detect = build_detector(ws, stack_repos=["vllm-project/vllm",
                                             "sgl-project/sglang"])
    detect.poll([c.model_id for c in decl.classes])

    decisions, skipped = {}, []

    def resolve(w, _trig):
        try:
            rec = decide(workload_class=w.workload_class,
                         model_key=w.model_key, incumbent=w.current_silicon,
                         slo_bound_ms=w.slo_bound_ms, batch=w.batch,
                         prompt_tokens=w.prompt_tokens,
                         output_tokens=w.output_tokens)
        except SystemExit as e:
            skipped.append((w.workload_class, str(e)))
            from berth.agent import Decision
            return Decision(w.current_silicon, w.current_silicon, 0.0, 0.0,
                            1.0, 0, 0, True)
        decisions[w.workload_class] = rec
        return rec.to_decision()

    # Delivery, only when --live and a token is present. Without both, the
    # pass decides and records and touches nothing, which is the same shape
    # as a dry run in any tool that changes infrastructure.
    deliver = None
    if args.live:
        import os
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise SystemExit(
                "--live needs GITHUB_TOKEN. Without a credential this would "
                "decide and silently change nothing, which is worse than "
                "refusing: you would believe it ran.")
        if not args.repo:
            raise SystemExit("--live needs --repo owner/name")
        from berth.execute import ExecutionState
        from berth.execute import execute as do_execute
        from berth.github import GitHubClient, open_proposal
        client = GitHubClient(token)
        target = decl.repo_target(*args.repo.split("/", 1))
        exec_state = ExecutionState()

        def deliver(proposal, policy):
            from berth.execute import AutonomyPolicy
            return do_execute(client, target, proposal,
                              policy or AutonomyPolicy(), exec_state,
                              open_proposal_fn=open_proposal)

    result = run(decl.classes, resolve, detect, state=ags,
                 shadow=not args.live, deliver=deliver,
                 autonomy=decl.autonomy)

    mode = "LIVE" if args.live else "shadow"
    print(f"{mode}: {len(decl.classes)} classes, {result.triggers_seen} "
          f"triggers, {len(result.proposals)} proposals, "
          f"{result.moved} executed")
    if result.triggers_seen and result.proposal_rate < 0.25:
        print("  Below the 0.25 kill criterion. Sustained, the trigger set is "
              "wrong and the agent is noise rather than coverage.")
    for src, why in sorted(ws.unreachable.items()):
        print(f"  UNREACHABLE {src}: {why[:70]}")
    for cls, why in result.suppressed:
        print(f"  silent  {cls}: {why}")
    for cls, why in skipped:
        print(f"  SKIPPED {cls}: {why[:70]}")
    for p in result.proposals:
        print(f"\n  {p.title}")
        for line in p.config_diff.splitlines():
            print(f"    {line}")
    for r in getattr(result, "executions", []):
        verb = "EXECUTED" if r.executed else "held"
        print(f"\n  {verb}  {r.workload_class}: {r.reason}")

    if args.state:
        Path(args.state).write_text(_json.dumps({
            "model_versions": ws.model_versions,
            "stack_versions": ws.stack_versions,
            "prices": ws.prices,
            "corpus_cells": sorted(ws.corpus_cells)}, indent=2))
    return 0


def cmd_versus(args):
    """Should you self-host at all, and against which API.

    The question most teams actually have, and the one nothing else can
    answer: what a hosted endpoint charges is published, and what the same
    work would cost on hardware they would rent is not.
    """
    import json as _json
    from dataclasses import asdict

    from berth.versus import ApiOffer, compare, render
    with open(args.offers) as f:
        offers = [ApiOffer(**o) for o in _json.load(f)]
    c = compare(model_key=args.model, prompt_tokens=args.prompt,
                output_tokens=args.output, slo_bound_ms=args.slo_ms,
                requests_per_hour=args.requests_per_hour,
                api_offers=offers, concurrency=args.concurrency,
                engineering_cost_per_hour=args.engineering_per_hour)
    print(_json.dumps(asdict(c), indent=2, default=str) if args.json
          else render(c))
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

    pl = sub.add_parser("place", help="emit a placement decision record")
    pl.add_argument("--workload-class", required=True)
    pl.add_argument("--model", required=True)
    pl.add_argument("--incumbent", required=True,
                    help="the placement running today")
    pl.add_argument("--slo-ms", type=float, default=1000.0)
    pl.add_argument("--batch", type=int, default=8)
    pl.add_argument("--prompt", type=int, default=512)
    pl.add_argument("--output", type=int, default=128)
    pl.add_argument("--json", action="store_true")
    pl.set_defaults(func=cmd_place)

    vs = sub.add_parser("versus",
                        help="compare self-hosting against provider APIs")
    vs.add_argument("--model", required=True)
    vs.add_argument("--offers", required=True,
                    help="JSON list of API rate cards")
    vs.add_argument("--prompt", type=int, required=True)
    vs.add_argument("--output", type=int, required=True)
    vs.add_argument("--slo-ms", type=float, required=True)
    vs.add_argument("--requests-per-hour", type=float, required=True,
                    help="the load that decides it. A rented node is paid for "
                         "idle and an API is not.")
    vs.add_argument("--concurrency", type=int, default=8)
    vs.add_argument("--engineering-per-hour", type=float, default=0.0,
                    help="loaded cost of the time spent running it. At typical "
                         "volumes this is the larger term.")
    vs.add_argument("--json", action="store_true")
    vs.set_defaults(func=cmd_versus)

    pi = sub.add_parser("pilot",
                        help="run one pass of the placement agent")
    pi.add_argument("--classes", required=True,
                    help="JSON file of watched workload classes")
    pi.add_argument("--state", help="file to persist watcher state between runs")
    pi.add_argument("--repo", help="owner/name, required with --live")
    pi.add_argument("--live", action="store_true",
                    help="open pull requests and execute where the declared "
                         "autonomy policy permits. Needs GITHUB_TOKEN and "
                         "--repo. Default is shadow: decide, record, touch "
                         "nothing.")
    pi.set_defaults(func=cmd_pilot)

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
