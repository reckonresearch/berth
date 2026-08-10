#!/usr/bin/env python3
"""Run Pilot in shadow against our own corpus.

The roadmap says the agent earns the right to open a pull request by running
against our own corpus for a quarter first. This is that run.

Shadow means it decides everything and opens nothing. State persists between
passes, so the first pass records what every source looks like and fires
nothing, and every pass after it reports only what actually moved.

    python -m bench.pilot_shadow                    one pass
    python -m bench.pilot_shadow --status           also write .berth/STATUS.md
    python -m bench.pilot_shadow --summary          the kill-criterion numbers

Run it daily. A cron line is enough:

    0 9 * * *  cd /path/to/berth-run && python -m bench.pilot_shadow >> .berth/pilot.log 2>&1

What to read after a month. The proposal rate is the kill criterion: fewer
than one trigger in four producing a change that clears the confidence band
means the trigger set is wrong. The suppression reasons are more informative
than the proposals, because they say which sources are noisy.
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from berth.agent import AgentRun, AgentState, Outcome, ProposalRecord, run
from berth.declaration import load_yaml
from berth.place import decide
from berth.status import render_status
from berth.watch import WatchState, build_detector

STATE_PATH = Path(".berth/pilot-state.json")
LOG_PATH = Path(".berth/pilot-runs.jsonl")
CLASSES = Path(".berth/classes.yaml")

# Serving stacks worth watching. The version effect measured on one H100 SXM
# was 1.48x at batch 1 and 2.70x at batch 32 between vLLM 0.5.5 and 0.25,
# which is larger than most placement decisions this system makes.
STACK_REPOS = ["vllm-project/vllm", "sgl-project/sglang"]

# Cells the corpus has measured. A new entry here is a prior becoming a
# measurement, which is the only trigger where the answer can change without
# anything in the outside world changing.
CORPUS_CELLS = [
    ("l40s", "llama3-8b"),
    ("h100-pcie", "llama3-8b"),
    ("h100-sxm", "llama3-8b"),
    ("mi300x", "llama3-8b"),
    ("a100-80g", "qwen3-30b-a3b"),
]


def load_state():
    ws, ags = WatchState(), AgentState()
    if not STATE_PATH.exists():
        return ws, ags, True
    d = json.loads(STATE_PATH.read_text())
    ws.model_versions = d.get("model_versions", {})
    ws.stack_versions = d.get("stack_versions", {})
    ws.prices = d.get("prices", {})
    ws.corpus_cells = {tuple(c) for c in d.get("corpus_cells", [])}
    ws.last_polled = d.get("last_polled", {})
    ags.records = [ProposalRecord(**r) for r in d.get("records", [])]
    for r in ags.records:
        r.outcome = Outcome(r.outcome)
    return ws, ags, False


def save_state(ws: WatchState, ags: AgentState):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps({
        "model_versions": ws.model_versions,
        "stack_versions": ws.stack_versions,
        "prices": ws.prices,
        "corpus_cells": sorted(ws.corpus_cells),
        "unreachable": ws.unreachable,
        "last_polled": ws.last_polled,
        "records": [{**asdict(r), "outcome": str(r.outcome)} for r in ags.records],
    }, indent=2))


def log_run(result: AgentRun, first: bool, unreachable=None):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps({
            "at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "first_pass": first,
            "triggers": result.triggers_seen,
            "estimates": result.estimates_run,
            "proposals": [p.title for p in result.proposals],
            "suppressed": result.suppressed,
            "unreachable": dict(unreachable or {}),
        }) + "\n")


def summarise():
    if not LOG_PATH.exists():
        sys.exit("no runs yet")
    runs = [json.loads(ln) for ln in LOG_PATH.read_text().splitlines() if ln.strip()]
    trig = sum(r["triggers"] for r in runs)
    props = sum(len(r["proposals"]) for r in runs)
    print(f"{len(runs)} passes, {trig} triggers, {props} proposals")
    if trig:
        rate = props / trig
        print(f"proposal rate {rate:.0%}")
        if rate < 0.25:
            print("  Below the 0.25 kill criterion. Sustained, that means the "
                  "trigger set is wrong and the agent is noise rather than "
                  "coverage. Retune the triggers or drop the feature.")
        else:
            print("  Above the kill criterion.")
    dead = sum(1 for r in runs if r.get("unreachable"))
    if dead:
        print(f"{dead} of {len(runs)} passes had at least one unreachable "
              f"source. A pass that reached nothing is not evidence that "
              f"nothing changed, and the rate above is computed over all of "
              f"them.")
    reasons = {}
    for r in runs:
        for _cls, why in r["suppressed"]:
            key = why.split(",")[0].split(".")[0][:60]
            reasons[key] = reasons.get(key, 0) + 1
    if reasons:
        print("\nwhy it stayed silent, which is the more informative half:")
        for why, n in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {n:>4}  {why}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m bench.pilot_shadow")
    p.add_argument("--status", action="store_true",
                   help="also write .berth/STATUS.md")
    p.add_argument("--summary", action="store_true",
                   help="print the kill-criterion numbers and exit")
    args = p.parse_args(argv)

    if args.summary:
        summarise()
        return 0

    if not CLASSES.exists():
        sys.exit(f"no {CLASSES}. Declare what to watch first.")
    decl = load_yaml(CLASSES.read_text(), repo="reckon-research/berth")
    ws, ags, first = load_state()

    detect = build_detector(ws, corpus_cells=CORPUS_CELLS,
                            stack_repos=STACK_REPOS)
    detect.poll([c.model_id for c in decl.classes])

    decisions = {}

    skipped = []

    def resolve(w, _trig):
        # A declaration naming a model this build does not know is a gap in
        # the registry, not grounds for abandoning every other class. The pass
        # continues and says which ones it could not evaluate, because a run
        # that dies on the fourth of four classes reports nothing about the
        # first three.
        try:
            rec = decide(workload_class=w.workload_class,
                         model_key=w.model_key,
                         incumbent=w.current_silicon,
                         slo_bound_ms=w.slo_bound_ms, batch=w.batch,
                         prompt_tokens=w.prompt_tokens,
                         output_tokens=w.output_tokens)
        except SystemExit as e:
            skipped.append((w.workload_class, str(e)))
            return None
        decisions[w.workload_class] = rec
        return rec.to_decision()

    def resolve_guarded(w, trig):
        d = resolve(w, trig)
        if d is None:
            # Nothing to compare, so nothing to propose.
            from berth.agent import Decision
            return Decision(recommended=w.current_silicon,
                            incumbent=w.current_silicon,
                            recommended_cost=0.0, incumbent_cost=0.0,
                            confidence_band=1.0, measured_cells=0,
                            prior_cells=0, feasible=True)
        return d

    result = run(decl.classes, resolve_guarded, detect, state=ags, shadow=True)

    print(f"shadow pass, {len(decl.classes)} classes")
    if first:
        print("  First pass. Every source is recorded and nothing fires, "
              "because we do not yet know whether anything changed.")
    for src, when in sorted(ws.last_polled.items()):
        print(f"  polled  {src:<34} {when[:16].replace('T', ' ')}")
    for src, why in sorted(ws.unreachable.items()):
        print(f"  UNREACHABLE  {src:<29} {why}")

    # A pass where nothing could be reached is not a quiet pass. Saying so is
    # the difference between the log being evidence and being noise, and the
    # first run of this system had every remote source dead behind an SSL
    # failure while the output read as normal.
    remote = [s for s in ws.last_polled if s.startswith(("model:", "stack:"))]
    if ws.unreachable and not remote:
        print("\n  NO REMOTE SOURCE WAS REACHED on this pass. Everything below "
              "comes from the local corpus alone, so an absence of triggers "
              "here means nothing about whether the world moved.")
    elif ws.unreachable:
        print(f"\n  {len(ws.unreachable)} source(s) unreachable. Triggers below "
              f"are from the sources that did answer.")
    print(f"\n  triggers {result.triggers_seen}   proposals "
          f"{len(result.proposals)}   suppressed {len(result.suppressed)}")
    for cls, why in result.suppressed:
        print(f"    silent  {cls}: {why}")
    for cls, why in skipped:
        print(f"    SKIPPED {cls}: {why}")
    for prop in result.proposals:
        print(f"\n  WOULD PROPOSE: {prop.title}")
        for line in prop.config_diff.splitlines():
            print(f"    {line}")

    if args.status:
        # Every class needs a decision for the page, even those no trigger
        # touched, or the reader sees a gap and cannot tell it from a failure.
        for c in decl.classes:
            if c.workload_class not in decisions:
                resolve(c, None)
        # Classes that could not be evaluated are absent from `decisions`, and
        # the status page renders them as "not evaluated" rather than omitting
        # them, so a gap is visible rather than silent.
        md = render_status(classes=decl.classes, decisions=decisions,
                           agent_state=ags, watch_state=ws,
                           repo="reckon-research/berth", shadow=True)
        Path(".berth/STATUS.md").write_text(md)
        print("\n  wrote .berth/STATUS.md")

    save_state(ws, ags)
    log_run(result, first, ws.unreachable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
