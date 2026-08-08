"""One place to see everything, and it is a file in their repository.

At one workload class the distributed model is fine: the placement is in the
deployment config, the reason is in the pull request, the history is in the
git log. At twenty classes across three repositories, "what is the state of my
placements" has no answer, and the assurance product is priced per class and
expands by class, so twenty is the expected state rather than the edge.

That fragmentation is real and it needs solving. A dashboard solves it and
brings its own costs: authentication, hosting, session state, and a second
source of truth about what is deployed that will eventually disagree with the
first.

**The unification needed is a read surface, not a write surface.** So this
writes `.berth/STATUS.md` into the customer's repository on every pass:

- Every class, its current placement, and what it is holding
- The open proposal for each, if any, linked
- What was last merged and what it saved
- Which cells the recommendations rest on, measured or prior
- When each source was last polled

One place to look. Version controlled, so its history is the history of the
account. Renders natively wherever the repository is browsed. No login, no
hosting, no second source of truth, and deleting it is how you turn it off.
"""

from __future__ import annotations

from datetime import UTC, datetime

from berth.agent import AgentState, Outcome


def _billable(receipt: dict) -> str:
    if receipt.get("billable"):
        return "yes"
    return "no, " + (receipt.get("reason_not_billable") or "")[:40]


def _sym(outcome: Outcome) -> str:
    return {Outcome.OPEN: "open",
            Outcome.MERGED: "merged",
            Outcome.REJECTED: "declined",
            Outcome.SUPERSEDED: "superseded",
            Outcome.SHADOW: "shadow"}.get(outcome, str(outcome))


def render_status(*, classes, decisions, agent_state: AgentState,
                  watch_state=None, receipts=None, repo: str = "",
                  shadow: bool = False, pr_url_for=None) -> str:
    """Compose the status page.

    `decisions` maps workload class to the current PlacementRecord, so the
    page shows what the estimator says now rather than what it said when the
    last proposal was opened. Those differ whenever the corpus has moved, and
    the difference is the most useful thing on the page.
    """
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    receipts = receipts or []
    out = [
        "# Placement status",
        "",
        f"Generated {now} by berth pilot"
        + (" in shadow mode, nothing is being proposed" if shadow else "")
        + ".",
        "",
        "This file is written by the agent and overwritten on every pass. "
        "Delete it to stop, or remove the class from `.berth/classes.yaml` to "
        "stop watching one.",
        "",
        "## Classes",
        "",
        "| class | running on | holding | recommended now | basis | proposal |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for c in classes:
        d = decisions.get(c.workload_class)
        rec = open_rec = None
        for r in agent_state.records:
            if r.workload_class == c.workload_class and r.outcome == Outcome.OPEN:
                open_rec = r
            if r.workload_class == c.workload_class and r.outcome == Outcome.MERGED:
                rec = r

        holding = f"{c.slo_metric} < {c.slo_bound_ms:.0f} ms"
        if d is None:
            out.append(f"| `{c.workload_class}` | {c.current_silicon} | "
                       f"{holding} | not evaluated | | |")
            continue

        if d.recommended == c.current_silicon:
            recommendation = "no change"
        elif not d.clears_band:
            recommendation = (f"{d.recommended}, {d.margin:.0%} inside a "
                              f"±{d.band:.0%} band")
        else:
            recommendation = f"**{d.recommended}**, {d.margin:.0%} better"

        basis = (f"{d.measured_cells} measured, {d.prior_cells} prior"
                 if d.measured_cells else f"{d.prior_cells} prior, unmeasured")

        if open_rec:
            link = pr_url_for(open_rec) if pr_url_for else ""
            proposal = f"[open]({link})" if link else "open"
        elif rec:
            proposal = f"merged {rec.outcome_utc[:10]}"
        else:
            proposal = ""

        out.append(f"| `{c.workload_class}` | {c.current_silicon} | {holding} "
                   f"| {recommendation} | {basis} | {proposal} |")

    # A recommendation resting on nothing is the single most important thing a
    # reader can know, so it is called out rather than left in a table column.
    unmeasured = [c.workload_class for c in classes
                  if (d := decisions.get(c.workload_class))
                  and d.clears_band and not d.measured_cells]
    if unmeasured:
        out += [
            "",
            "> **Unmeasured recommendations.** " + ", ".join(
                f"`{n}`" for n in unmeasured) + " would move to hardware "
            "nobody has measured for this model. These rest on spec-sheet "
            "arithmetic, and a prior has been observed above 40 percent error. "
            "Measuring the cell costs about four dollars and two hours.",
        ]

    if receipts:
        out += ["", "## Settled periods", "",
                "| period | class | net saving | invoiced | billable |",
                "| --- | --- | --- | --- | --- |"]
        for r in receipts:
            s = r.get("settlement", {})
            d = r.get("declaration", {})
            out.append(
                f"| {r.get('created_utc', '')[:10]} | "
                f"`{d.get('workload_class', '')}` | "
                f"{s.get('net_saving', 0):,.0f} | "
                f"{s.get('invoice_conservative', 0):,.0f} | "
                f"{_billable(r)} |")

    out += ["", "## Proposal history", ""]
    if not agent_state.records:
        out.append("Nothing proposed yet.")
    else:
        out += ["| opened | class | change | trigger | outcome |",
                "| --- | --- | --- | --- | --- |"]
        for r in sorted(agent_state.records,
                        key=lambda r: r.opened_utc, reverse=True)[:25]:
            out.append(f"| {r.opened_utc[:10]} | `{r.workload_class}` | "
                       f"{r.from_placement} to {r.to_placement} | {r.trigger} | "
                       f"{_sym(r.outcome)}{(' , ' + r.note) if r.note else ''} |")
        if agent_state.merge_rate:
            out += ["", f"Merge rate {agent_state.merge_rate:.0%} of resolved "
                        f"proposals. Consistently declined proposals are "
                        f"correct arithmetic nobody wants, which is worth a "
                        f"conversation rather than a threshold change."]

    if watch_state is not None and watch_state.last_polled:
        out += ["", "## Sources", "", "| source | last polled |",
                "| --- | --- |"]
        for src, when in sorted(watch_state.last_polled.items()):
            out.append(f"| {src} | {when[:16].replace('T', ' ')} |")

    out += [
        "",
        "---",
        "",
        "Every figure above is reproducible. `berth place --workload-class "
        "<name> --model <key> --incumbent <silicon>` regenerates any "
        "recommendation, and the traces behind each measured cell are at "
        "docs.reckonresearch.com.",
    ]
    return "\n".join(out) + "\n"


STATUS_PATH = ".berth/STATUS.md"


def publish(client, repo_target, content: str, *, branch: str,
            message: str = "chore: placement status"):
    """Write the status page on the proposal branch.

    Deliberately on the branch rather than direct to trunk: the same review
    that accepts a placement accepts the status update, and nothing this
    integration does ever lands without a human merging it.
    """
    target = repo_target
    if STATUS_PATH not in target.allowed_paths:
        target = type(repo_target)(
            owner=repo_target.owner, name=repo_target.name,
            default_branch=repo_target.default_branch,
            allowed_paths=tuple(repo_target.allowed_paths) + (STATUS_PATH,))
    try:
        _existing, sha = client.read_file(target, STATUS_PATH, ref=branch)
    except Exception:                      # noqa: BLE001 - absent is the normal case
        sha = None
    return client.put_file(target, STATUS_PATH, content, branch=branch,
                           sha=sha, message=message)
