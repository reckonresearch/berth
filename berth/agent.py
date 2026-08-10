"""The placement agent. Watches for change, re-estimates, opens a pull request.

A placement is not a decision, it is a position that decays. A model version
ships, a provider moves a price, a cell lands in the corpus, and the answer
that was right last month is not right this month. Nobody checks, because
checking requires re-measuring and nothing tells you when it is worth doing.

This is the loop that tells you.

    watch     model registries, price sources, corpus additions
    detect    a change touching a declared workload class
    estimate  re-run the placement decision against the new inputs
    propose   a pull request against the customer's deployment config,
              carrying the diff and the evidence

**Why a pull request rather than an alert.** An alert creates work. A diff with
evidence attached is work already done, arriving in a review workflow every
infrastructure team already has, where it can be discussed, amended, rejected
or merged. It also leaves the decision in the customer's own history rather
than in a vendor dashboard, and it can be reverted by reverting the commit.

**What it never does.** Merge. Deploy. Touch anything outside the declared
configuration paths. It needs read access to a config repository and write
access to a branch, and specifically not access to traffic, credentials, or
the request path.

**The rule that keeps it useful.** If the new answer is the same, or the margin
sits inside the confidence band, it stops silently. An agent that opens a pull
request every week is an agent that gets muted, and a muted agent is worse
than none because it looks like coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class Trigger(StrEnum):
    """Why the agent woke up. Recorded on the proposal, because the reason a
    placement changed is part of the decision."""

    MODEL_VERSION = "model_version"      # a declared model shipped an update
    PRICE_CHANGE = "price_change"        # a provider moved a rate
    CORPUS_CELL = "corpus_cell"          # a measurement replaced a prior
    SLO_CHANGE = "slo_change"            # the customer changed the bound
    STACK_VERSION = "stack_version"      # the serving stack cut a release
    SCHEDULED = "scheduled"              # periodic re-check, no external event


@dataclass(frozen=True)
class WatchedClass:
    """One workload class the agent is holding a position on."""

    workload_class: str
    model_id: str                        # what to watch in the registry
    model_key: str                       # the berth registry key
    current_silicon: str
    current_model_version: str
    slo_metric: str
    slo_bound_ms: float
    batch: int
    prompt_tokens: int
    output_tokens: int
    config_path: str                     # file in the customer's repo
    repo: str                            # owner/name


@dataclass
class Decision:
    """The output of one re-estimation, before it becomes a proposal."""

    recommended: str
    incumbent: str
    recommended_cost: float
    incumbent_cost: float
    confidence_band: float               # fractional, e.g. 0.106
    measured_cells: int
    prior_cells: int
    feasible: bool
    reason_infeasible: str | None = None
    # Cost of making the move, as a fraction of the incumbent's cost over the
    # period the decision is expected to hold. Redeployment, cache warm-up on
    # the new placement, and the risk that a move made this week is reversed
    # next week.
    #
    # Without it the loop is an oscillator waiting for a price feed. Two
    # placements a few percent apart will trade the lead every time a rate
    # card moves, and each swap costs real money while the estimate that
    # justified it was never wrong. Damping is a property of adaptation
    # itself, not a policy laid over it, which is why it lives in the decision
    # rather than in a threshold somewhere upstream.
    switching_cost: float = 0.03

    @property
    def margin(self) -> float:
        """Fractional improvement over the incumbent."""
        if self.incumbent_cost <= 0:
            return 0.0
        return (self.incumbent_cost - self.recommended_cost) / self.incumbent_cost

    @property
    def hurdle(self) -> float:
        """What the gain must beat: the uncertainty plus the cost of moving.

        Two terms because they fail differently. A gain inside the confidence
        band might not exist. A gain smaller than the switching cost exists
        and is not worth collecting.
        """
        return self.confidence_band + self.switching_cost

    @property
    def clears_band(self) -> bool:
        """Whether the improvement is worth acting on.

        The gate that decides whether anything is proposed. A recommendation
        inside the noise floor is not a recommendation, and one that costs
        more to enact than it returns is worse than silence because it looks
        like progress.
        """
        return self.margin > self.hurdle

    @property
    def rests_on_measurement(self) -> bool:
        """Whether the recommended placement has been measured at all.

        A proposal resting entirely on spec-sheet arithmetic is stated as
        such. A prior cell has been observed above 40 percent error in this
        corpus, and a customer deciding whether to move production traffic
        deserves to know which kind of number they are looking at.
        """
        return self.measured_cells > 0


@dataclass
class Proposal:
    """A pull request, before it is opened."""

    watched: WatchedClass
    trigger: Trigger
    decision: Decision
    config_diff: str
    title: str
    body: str
    created_utc: str = ""

    def __post_init__(self):
        if not self.created_utc:
            self.created_utc = datetime.now(UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ")


@dataclass
class AgentRun:
    """One pass of the loop, whether or not it proposed anything."""

    triggers_seen: int = 0
    estimates_run: int = 0
    proposals: list[Proposal] = field(default_factory=list)
    suppressed: list[tuple[str, str]] = field(default_factory=list)
    shadow: bool = False

    @property
    def proposal_rate(self) -> float:
        """Share of triggers that became a proposal.

        The kill criterion for this feature: if fewer than one trigger in four
        produces a placement change clearing the confidence band, the trigger
        set is wrong and the agent is noise. Retune it or drop it.
        """
        return len(self.proposals) / self.triggers_seen if self.triggers_seen else 0.0


# ------------------------------------------------------------------- state

class Outcome(StrEnum):
    """What became of a proposal. The agent is only useful if it remembers."""

    OPEN = "open"
    MERGED = "merged"
    REJECTED = "rejected"        # closed without merging
    SUPERSEDED = "superseded"    # a later proposal replaced it
    SHADOW = "shadow"            # generated but never opened


@dataclass
class ProposalRecord:
    """One proposal and what happened to it."""

    workload_class: str
    from_placement: str
    to_placement: str
    trigger: str
    opened_utc: str
    outcome: Outcome = Outcome.OPEN
    outcome_utc: str = ""
    note: str = ""


@dataclass
class AgentState:
    """What the agent knows about proposals it has already made.

    Without this the loop is memoryless: three triggers in a week produce
    three identical pull requests, and a customer who closed the first one
    gets it again the next day. An agent that repeats itself is an agent that
    gets muted, and a muted agent is worse than none because it looks like
    coverage.
    """

    records: list[ProposalRecord] = field(default_factory=list)
    # How long a rejection suppresses the same proposal. A customer who says
    # not this quarter means not this quarter, and re-asking is how the whole
    # channel gets muted. Re-proposing after this window is legitimate,
    # because the world moves.
    rejection_cooldown_days: int = 90

    def open_for(self, workload_class: str, to_placement: str):
        for r in self.records:
            if (r.workload_class == workload_class
                    and r.to_placement == to_placement
                    and r.outcome == Outcome.OPEN):
                return r
        return None

    def rejected_recently(self, workload_class: str, to_placement: str,
                          now: datetime) -> ProposalRecord | None:
        for r in self.records:
            if (r.workload_class == workload_class
                    and r.to_placement == to_placement
                    and r.outcome == Outcome.REJECTED and r.outcome_utc):
                age = (now - datetime.fromisoformat(
                    r.outcome_utc.replace("Z", "+00:00"))).days
                if age < self.rejection_cooldown_days:
                    return r
        return None

    def record(self, proposal: Proposal, outcome: Outcome = Outcome.OPEN):
        # A new proposal for the same class supersedes any open one, so the
        # customer never has two live pull requests pointing different ways.
        for r in self.records:
            if (r.workload_class == proposal.watched.workload_class
                    and r.outcome == Outcome.OPEN):
                r.outcome = Outcome.SUPERSEDED
                r.outcome_utc = proposal.created_utc
        self.records.append(ProposalRecord(
            workload_class=proposal.watched.workload_class,
            from_placement=proposal.decision.incumbent,
            to_placement=proposal.decision.recommended,
            trigger=str(proposal.trigger),
            opened_utc=proposal.created_utc,
            outcome=outcome))

    def resolve(self, workload_class: str, outcome: Outcome, note: str = "",
                when: str = ""):
        """Record what a customer did. Called by whatever watches the repo."""
        for r in self.records:
            if r.workload_class == workload_class and r.outcome == Outcome.OPEN:
                r.outcome = outcome
                r.outcome_utc = when or datetime.now(UTC).strftime(
                    "%Y-%m-%dT%H:%M:%SZ")
                r.note = note
                return r
        return None

    @property
    def merge_rate(self) -> float:
        """Share of resolved proposals the customer merged.

        The second kill criterion, and a better one than the proposal rate:
        an agent whose proposals are consistently rejected is producing
        correct arithmetic that nobody wants, which is a product problem
        rather than a threshold problem.
        """
        resolved = [r for r in self.records
                    if r.outcome in (Outcome.MERGED, Outcome.REJECTED)]
        if not resolved:
            return 0.0
        return sum(1 for r in resolved if r.outcome == Outcome.MERGED) / len(resolved)


# ------------------------------------------------------------------- the loop

def evaluate(watched: WatchedClass, trigger: Trigger, decision: Decision,
             state: AgentState | None = None,
             now: datetime | None = None) -> tuple[bool, str]:
    """Whether this decision should become a pull request.

    Returns (propose, reason). The reason is recorded either way, because a
    suppressed trigger is evidence about the trigger set and the kill
    criterion is computed from it.
    """
    now = now or datetime.now(UTC)
    if state is not None:
        already = state.open_for(watched.workload_class, decision.recommended)
        if already:
            return False, (f"the same proposal is already open, from "
                           f"{already.opened_utc}. Repeating it is how the "
                           f"channel gets muted")
        rejected = state.rejected_recently(watched.workload_class,
                                           decision.recommended, now)
        if rejected:
            return False, (f"the customer rejected this move on "
                           f"{rejected.outcome_utc[:10]} and the cooldown has "
                           f"not elapsed. A rejection is an answer")

    if not decision.feasible:
        # Worth proposing: an infeasible incumbent is a problem the customer
        # has whether or not a better placement exists.
        if decision.recommended == decision.incumbent:
            return False, ("no feasible alternative and the incumbent is "
                           "infeasible, which is a capacity problem rather "
                           "than a placement one")
        return True, "the incumbent cannot serve this workload"

    if decision.recommended == decision.incumbent:
        return False, "the answer did not change"

    if not decision.clears_band:
        if decision.margin <= decision.confidence_band:
            return False, (f"improvement of {decision.margin:.1%} sits inside "
                           f"the confidence band of "
                           f"{decision.confidence_band:.1%}, so it is not "
                           f"distinguishable from noise")
        return False, (f"improvement of {decision.margin:.1%} clears the "
                       f"{decision.confidence_band:.1%} band but not the "
                       f"{decision.switching_cost:.1%} cost of moving. The "
                       f"gain is real and smaller than collecting it")

    return True, (f"{decision.margin:.1%} improvement against a "
                  f"{decision.confidence_band:.1%} band")


def build_proposal(watched: WatchedClass, trigger: Trigger, decision: Decision,
                   *, config_diff: str, traces_url: str) -> Proposal:
    """Compose the pull request.

    The body carries the evidence rather than a summary of it: what changed,
    what the estimate says, how much of it rests on measurement, and where the
    traces are. A reviewer should be able to disagree with the recommendation
    from the body alone.
    """
    d = decision
    saving = f"{d.margin:.1%}"
    basis = (f"{d.measured_cells} measured, {d.prior_cells} prior"
             if d.measured_cells else
             f"{d.prior_cells} prior, none measured")

    title = (f"placement: move {watched.workload_class} from "
             f"{d.incumbent} to {d.recommended}")

    lines = [
        f"Triggered by: **{trigger.value}**",
        "",
        f"`{watched.workload_class}` is running on **{d.incumbent}**. After "
        f"the change above, the cheaper placement under the declared bound is "
        f"**{d.recommended}**.",
        "",
        "| | incumbent | recommended |",
        "| --- | --- | --- |",
        f"| placement | {d.incumbent} | {d.recommended} |",
        f"| cost per Mtok | ${d.incumbent_cost:.3f} | ${d.recommended_cost:.3f} |",
        f"| improvement | | {saving} |",
        f"| confidence band | | plus or minus {d.confidence_band:.1%} |",
        f"| cost of moving | | {d.switching_cost:.1%} |",
        f"| hurdle cleared | | {d.margin:.1%} against {d.hurdle:.1%} |",
        "",
        f"Bound held: {watched.slo_metric} under {watched.slo_bound_ms:.0f} ms "
        f"at concurrency {watched.batch}, {watched.prompt_tokens} prompt "
        f"tokens, {watched.output_tokens} output tokens.",
        "",
        f"**Basis:** {basis}.",
    ]

    if not d.rests_on_measurement:
        lines += [
            "",
            "> This recommendation rests entirely on spec-sheet arithmetic. "
            "No cell in the corpus covers this placement, and a prior has been "
            "observed above 40 percent error. Treat it as a hypothesis worth "
            "measuring rather than a change worth making, or ask us to "
            "measure the cell first.",
        ]

    if not d.feasible:
        lines += ["", f"> **The incumbent cannot serve this workload.** "
                      f"{d.reason_infeasible}"]

    lines += [
        "",
        f"Traces and method: {traces_url}",
        "",
        "Reverting this is reverting the commit. Nothing outside the "
        "configuration paths in this diff was touched, and no traffic was "
        "read to produce it.",
    ]

    return Proposal(watched=watched, trigger=trigger, decision=decision,
                    config_diff=config_diff, title=title,
                    body="\n".join(lines))


def run(watched_classes, resolve_decision, detect_triggers,
        *, traces_url="https://docs.reckonresearch.com/validation-p0/",
        render_diff=None, state: AgentState | None = None,
        shadow: bool = False, now: datetime | None = None) -> AgentRun:
    """One pass over every class the agent is holding.

    `resolve_decision(watched, trigger) -> Decision` and
    `detect_triggers(watched) -> list[Trigger]` are injected, so the loop can
    be tested without a registry, a price feed, or a repository, and so a
    customer can run it against their own sources.
    """
    result = AgentRun()
    seen_this_run = set()
    for w in watched_classes:
        for trig in detect_triggers(w):
            result.triggers_seen += 1
            decision = resolve_decision(w, trig)
            result.estimates_run += 1
            # Several triggers in one pass can reach the same conclusion. The
            # first one proposes; the rest would be duplicates within a single
            # run, which no amount of persisted state would catch.
            key = (w.workload_class, decision.recommended)
            if key in seen_this_run:
                result.suppressed.append(
                    (w.workload_class,
                     "an earlier trigger in this run already reached this "
                     "conclusion"))
                continue
            propose, reason = evaluate(w, trig, decision, state, now)
            if not propose:
                result.suppressed.append((w.workload_class, reason))
                continue
            diff = (render_diff(w, decision) if render_diff
                    else default_diff(w, decision))
            proposal = build_proposal(w, trig, decision, config_diff=diff,
                                      traces_url=traces_url)
            seen_this_run.add(key)
            result.proposals.append(proposal)
            if state is not None:
                # Shadow mode records without opening anything, which is how
                # the agent earns the right to be trusted with a pull request:
                # run it against your own corpus for a quarter and read what
                # it would have said.
                state.record(proposal,
                             Outcome.SHADOW if shadow else Outcome.OPEN)
    if shadow:
        result.shadow = True
    return result


def default_diff(watched: WatchedClass, decision: Decision) -> str:
    """A unified diff over the declared configuration path.

    Deliberately minimal: one field. An agent that rewrites a deployment file
    is an agent nobody merges, and the smallest reviewable change is the one
    most likely to be accepted.
    """
    return "\n".join([
        f"--- a/{watched.config_path}",
        f"+++ b/{watched.config_path}",
        "@@",
        f"-  accelerator: {decision.incumbent}",
        f"+  accelerator: {decision.recommended}",
    ])
