"""The ledger. What running this has been worth, and what it has not.

An infra team needs to see three things: that it is running, that it is
working, and what it has been worth. The third is where a tool like this
usually starts lying, so the arithmetic here is arranged to make lying hard.

**Three quantities, never summed.**

*Realized* is a proposal that was merged, accruing from the moment it merged.
The workload moved and the cheaper placement is running. This is the number
that counts, and until a holdout period settles it, it is an estimate of
realized saving rather than a measurement of one.

*Available* is a proposal that is open and unmerged. Nobody has saved
anything. It is money on the table, and reporting it as saved would be the
single most tempting lie this system could tell.

*Foregone* is a proposal that was declined. Also not a loss: the customer had
a reason, usually one that has nothing to do with cost. It is reported because
a constraint has a price and the price should be visible to whoever set it.

**What is never counted.** A class the agent checked and left alone saved
nothing. The gap between its current placement and the worst placement in the
fleet is not a saving, it is a comparison nobody was going to make. Systems
that report that number are reporting their own existence as value.

**Provenance.** Every figure is ESTIMATED unless a receipt settled the period,
in which case it is VERIFIED and the receipt is named. The distinction is on
the face of the report, because an estimate presented as a measurement is the
one unrecoverable error in this project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from berth.agent import AgentState, Outcome

HOURS = {"run": None, "day": 24, "week": 168, "month": 730, "quarter": 2190}


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


@dataclass
class ClassEconomics:
    """What a class is worth per hour, declared by the customer.

    Without a volume the ledger cannot exist: a 20 percent improvement on a
    workload is a percentage until somebody says how much work there is. This
    is a declared quantity like the service level, not something the estimator
    infers, because inferring it would let the reported saving be adjusted by
    changing an assumption.
    """

    workload_class: str
    mtok_per_hour: float          # compliant output, million tokens
    incumbent_cost_per_mtok: float


@dataclass
class Entry:
    """One merged, open or declined proposal, with what it is worth."""

    workload_class: str
    from_placement: str
    to_placement: str
    state: str                    # realized | available | foregone
    opened: str
    settled: str | None
    delta_per_mtok: float
    mtok_per_hour: float
    hours_accrued: float
    amount: float
    provenance: str               # ESTIMATED | VERIFIED
    receipt: str | None = None
    note: str = ""


@dataclass
class Ledger:
    entries: list[Entry] = field(default_factory=list)
    as_of: str = ""

    def _sum(self, state: str, since_hours: float | None = None) -> float:
        now = _parse(self.as_of)
        total = 0.0
        for e in self.entries:
            if e.state != state:
                continue
            if since_hours is None:
                total += e.amount
                continue
            # Accrual inside a window, not the whole entry. A change merged in
            # March did not earn its March money again in August, and a report
            # that says otherwise is counting the same dollar every period.
            start = max(_parse(e.opened), now - timedelta(hours=since_hours))
            hours = max(0.0, (now - start).total_seconds() / 3600.0)
            total += e.delta_per_mtok * e.mtok_per_hour * hours
        return total

    def realized(self, window: str | None = None) -> float:
        return self._sum("realized", HOURS.get(window) if window else None)

    def available(self) -> float:
        """Per hour, not cumulative. Money on the table accrues to nobody, and
        stating it as a total would make waiting look like earning."""
        return sum(e.delta_per_mtok * e.mtok_per_hour
                   for e in self.entries if e.state == "available")

    def foregone(self) -> float:
        return sum(e.delta_per_mtok * e.mtok_per_hour
                   for e in self.entries if e.state == "foregone")

    @property
    def verified_share(self) -> float:
        """Share of realized money that a settled receipt stands behind.

        The number that says how much of this report is measurement and how
        much is arithmetic over an estimate. It starts at zero and should be
        stated as zero rather than omitted.
        """
        real = [e for e in self.entries if e.state == "realized"]
        if not real:
            return 0.0
        tot = sum(e.amount for e in real)
        ver = sum(e.amount for e in real if e.provenance == "VERIFIED")
        return ver / tot if tot else 0.0


def build_ledger(agent_state: AgentState, economics, *, decisions=None,
                 receipts=None, as_of: str | None = None) -> Ledger:
    """Turn proposal history into a ledger.

    `economics` maps workload class to ClassEconomics. A class without one is
    skipped and named, rather than defaulted to a volume nobody declared.
    """
    now = as_of or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    econ = {e.workload_class: e for e in economics}
    by_receipt = {r.get("declaration", {}).get("workload_class"): r
                  for r in (receipts or [])}
    decisions = decisions or {}

    entries = []
    for r in agent_state.records:
        e = econ.get(r.workload_class)
        if e is None:
            continue
        d = decisions.get(r.workload_class)
        margin = getattr(d, "margin", None)
        if margin is None:
            continue
        delta = margin * e.incumbent_cost_per_mtok

        if r.outcome == Outcome.MERGED and r.outcome_utc:
            hours = (_parse(now) - _parse(r.outcome_utc)).total_seconds() / 3600.0
            rec = by_receipt.get(r.workload_class)
            verified = bool(rec and rec.get("billable"))
            entries.append(Entry(
                workload_class=r.workload_class,
                from_placement=r.from_placement, to_placement=r.to_placement,
                state="realized", opened=r.outcome_utc, settled=None,
                delta_per_mtok=delta, mtok_per_hour=e.mtok_per_hour,
                hours_accrued=max(0.0, hours),
                amount=delta * e.mtok_per_hour * max(0.0, hours),
                provenance="VERIFIED" if verified else "ESTIMATED",
                receipt=(rec or {}).get("trace_pointer") if verified else None))
        elif r.outcome == Outcome.OPEN:
            entries.append(Entry(
                workload_class=r.workload_class,
                from_placement=r.from_placement, to_placement=r.to_placement,
                state="available", opened=r.opened_utc, settled=None,
                delta_per_mtok=delta, mtok_per_hour=e.mtok_per_hour,
                hours_accrued=0.0, amount=0.0, provenance="ESTIMATED",
                note="open, unmerged. Nothing has been saved."))
        elif r.outcome == Outcome.REJECTED:
            entries.append(Entry(
                workload_class=r.workload_class,
                from_placement=r.from_placement, to_placement=r.to_placement,
                state="foregone", opened=r.opened_utc, settled=r.outcome_utc,
                delta_per_mtok=delta, mtok_per_hour=e.mtok_per_hour,
                hours_accrued=0.0, amount=0.0, provenance="ESTIMATED",
                note=r.note or "declined"))
    return Ledger(entries=entries, as_of=now)


# ------------------------------------------------------------------ report

def daily_report(*, ledger: Ledger, classes, unreachable=None, triggers=0,
                 proposals=0, sources_polled=0, skipped=None) -> str:
    """The line that lands every morning.

    Its first job is not the money. It is to say what was checked, because a
    team that hears nothing for three weeks does not conclude all is well,
    they conclude it stopped running. Silence is correct for action and wrong
    for communication, and those are different things.
    """
    unreachable = unreachable or {}
    skipped = skipped or []
    n = len(classes)

    if unreachable and not sources_polled:
        head = ("PILOT DID NOT RUN CLEANLY. No source could be reached, so "
                "nothing below is evidence that nothing changed.")
    elif unreachable:
        head = (f"Checked {n} workload class{'es' if n != 1 else ''} against "
                f"{sources_polled} sources. {len(unreachable)} source(s) "
                f"unreachable.")
    elif proposals == 0:
        head = (f"Checked {n} workload class{'es' if n != 1 else ''} against "
                f"{sources_polled} sources. Every one is holding its service "
                f"level on the cheapest placement we can find. Nothing to do.")
    else:
        head = (f"Checked {n} workload class{'es' if n != 1 else ''} against "
                f"{sources_polled} sources. {proposals} placement"
                f"{'s' if proposals != 1 else ''} changed and "
                f"{'a pull request is' if proposals == 1 else 'pull requests are'} "
                f"open.")

    out = [f"PILOT  {ledger.as_of[:10]}", "=" * 64, "", head, ""]

    for src, why in sorted(unreachable.items()):
        out.append(f"  unreachable  {src}: {why[:70]}")
    for cls, why in skipped:
        out.append(f"  skipped      {cls}: {why[:70]}")
    if unreachable or skipped:
        out.append("")

    realized_total = ledger.realized()
    if realized_total or ledger.available():
        out += ["  SAVINGS", "  " + "-" * 62]
    if realized_total:
        out += [
            f"  realized, running total       ${realized_total:>12,.0f}",
            f"    last 24 hours               ${ledger.realized('day'):>12,.0f}",
            f"    last 7 days                 ${ledger.realized('week'):>12,.0f}",
            f"    last 30 days                ${ledger.realized('month'):>12,.0f}",
            f"    last quarter                ${ledger.realized('quarter'):>12,.0f}",
        ]
        v = ledger.verified_share
        out.append(
            f"  {v:.0%} of that is VERIFIED by a settled holdout period. The "
            f"rest is ESTIMATED: the placement moved and the estimate says "
            f"what it is worth, but no holdout has proved it."
            if v else
            "  0% of that is VERIFIED. Every figure above is ESTIMATED: the "
            "placements moved and the model says what they are worth, and no "
            "holdout period has proved it yet.")
        out.append("")

    if ledger.available():
        out += [
            f"  available, not yet taken      ${ledger.available():>12,.2f} per hour",
            "  Open pull requests that nobody has merged. Nothing has been "
            "saved here, and it accrues to no one while it waits.",
            "",
        ]

    if ledger.foregone():
        out += [
            f"  foregone by choice            ${ledger.foregone():>12,.2f} per hour",
            "  Proposals that were declined. Not a loss, and not free: this is "
            "what the constraints behind those decisions cost per hour.",
            "",
        ]

    if not realized_total and not ledger.available():
        out += [
            "  SAVINGS",
            "  " + "-" * 62,
            "  Nothing yet. No proposal has been merged, so no placement has "
            "moved and nothing has been saved. A class we checked and left "
            "alone saved nothing either, and reporting the gap to the worst "
            "placement in the fleet as a saving would be reporting our own "
            "existence as value.",
            "",
        ]

    out += [
        "  " + "-" * 62,
        f"  {triggers} trigger{'s' if triggers != 1 else ''} evaluated. "
        f"{proposals} cleared the confidence band and the cost of moving.",
        "  Every figure is reproducible: berth place --workload-class <name>.",
    ]
    return "\n".join(out)
