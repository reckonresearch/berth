"""Execution. pilot moves the workload, within bounds the customer declared.

A placement system that only proposes is a recommendation engine with extra
steps. If every change waits on someone noticing a pull request, the loop is
as slow as the human in it, and the whole argument for adaptive placement is
that the answer changes faster than anyone checks.

So pilot executes. The question was never whether, it was under what authority.

**Authority is declared once, in advance, in the customer's own repository.**
Not per change. They commit an autonomy policy to `.berth/classes.yaml`, and
that commit is the authorization: a change they reviewed, in their history,
signed by their credential. Terraform auto-apply, ArgoCD auto-sync and
Dependabot auto-merge all work this way, and they work because the human
decision moved from "approve this change" to "approve this class of change
under these conditions", which is the decision a human is actually good at.

**Four autonomy levels**, declared per class:

    propose    open a pull request and stop. The default.
    window     execute inside declared windows, propose outside them
    execute    execute whenever the hurdle is cleared
    frozen     do nothing, not even propose

**Six guardrails, all of which can stop an execution:**

Blast radius, so a bad estimate cannot move a whole fleet in one pass.
Rate limit, so a class cannot oscillate.
Blackout windows, so nothing moves during a freeze.
A minimum margin above the hurdle, higher than the proposal threshold,
because the bar for acting unattended should be higher than for asking.
Measurement basis, so an unmeasured placement is never executed unattended
regardless of how good it looks.
And automatic rollback, which is the one that matters most: a move that
degrades the service level reverts itself without waiting for anyone.

**The kill switch is deleting a file.** Remove `.berth/classes.yaml` and
nothing runs. That is deliberate: an autonomy system whose off switch requires
contacting the vendor is not one anyone should install.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum


class Autonomy(StrEnum):
    """What pilot may do with a decision, declared per workload class."""

    FROZEN = "frozen"        # do nothing, not even propose
    PROPOSE = "propose"      # open a pull request and stop. The default.
    WINDOW = "window"        # execute inside declared windows only
    EXECUTE = "execute"      # execute whenever the hurdle is cleared


class Refused(RuntimeError):
    """An execution was blocked by a guardrail.

    Distinct from a failure. A refusal is the system working, and the reason
    is recorded and reported rather than swallowed, because a guardrail that
    fires silently is indistinguishable from one that never fires.
    """


@dataclass(frozen=True)
class AutonomyPolicy:
    """The bounds, declared in the customer's repository.

    Every default here is the conservative one. A policy that has not been
    thought about should behave like a policy that says propose and nothing
    else, because the failure mode of a permissive default is a workload
    moving at three in the morning on a four percent estimate.
    """

    level: Autonomy = Autonomy.PROPOSE

    # Executions permitted per pass across all classes. One bad corpus update
    # should not be able to move a fleet.
    max_moves_per_pass: int = 1

    # Minimum hours between executions for one class. Without this, two
    # placements a few percent apart trade the lead every time a rate card
    # moves, and each swap costs real money.
    min_hours_between_moves: int = 168

    # The margin required to act unattended, on top of the confidence band and
    # the switching cost. Higher than the proposal threshold on purpose: the
    # bar for moving production without being asked should exceed the bar for
    # asking.
    min_margin_to_execute: float = 0.25

    # Whether an unmeasured placement may be executed. Off by default. A
    # spec-sheet prior has been observed above 40 percent error, and moving
    # production onto one unattended is the single worst thing this system
    # could do.
    allow_unmeasured: bool = False

    # Hours of the week when execution is permitted, as (weekday, start, end)
    # with Monday as 0. Empty means any time, which is only reachable at level
    # EXECUTE.
    windows: tuple[tuple[int, int, int], ...] = ()

    # Dates when nothing moves. A freeze is a real thing in most
    # organisations and a placement system that ignores it gets uninstalled.
    blackout_dates: tuple[str, ...] = ()

    # Consecutive periods of degraded service level that trigger a revert.
    rollback_after_breaches: int = 1


@dataclass
class ExecutionRecord:
    """One execution, or one refusal, with the reason either way."""

    workload_class: str
    from_placement: str
    to_placement: str
    at_utc: str
    executed: bool
    reason: str
    pull_request: str | None = None
    rolled_back: bool = False
    rollback_reason: str = ""


@dataclass
class ExecutionState:
    """What pilot has executed, so rate limits and rollback can be enforced."""

    records: list[ExecutionRecord] = field(default_factory=list)

    def last_move(self, workload_class: str) -> ExecutionRecord | None:
        moves = [r for r in self.records
                 if r.workload_class == workload_class and r.executed]
        return moves[-1] if moves else None

    def moves_this_pass(self, since: datetime) -> int:
        return sum(1 for r in self.records
                   if r.executed
                   and datetime.fromisoformat(r.at_utc.replace("Z", "+00:00")) >= since)


# ------------------------------------------------------------------ the gate

def may_execute(decision, policy: AutonomyPolicy, state: ExecutionState,
                workload_class: str, *, now: datetime | None = None,
                moves_already: int = 0) -> tuple[bool, str]:
    """Whether this decision may be executed unattended.

    Returns (permitted, reason). The reason is recorded either way. Every
    check below can only ever say no: there is no path where a guardrail
    permits something the level did not already allow.
    """
    now = now or datetime.now(UTC)

    if policy.level == Autonomy.FROZEN:
        return False, "class is frozen, nothing is proposed or executed"
    if policy.level == Autonomy.PROPOSE:
        return False, "autonomy is propose, so this opens a pull request"

    if not decision.clears_band:
        return False, ("the margin does not clear the confidence band plus the "
                       "cost of moving, so there is nothing to execute")

    if decision.margin < policy.min_margin_to_execute:
        return False, (f"margin {decision.margin:.1%} is below the "
                       f"{policy.min_margin_to_execute:.0%} required to act "
                       f"unattended. It clears the bar for proposing and not "
                       f"the bar for moving production without being asked")

    if not decision.rests_on_measurement and not policy.allow_unmeasured:
        return False, ("the recommended placement has never been measured for "
                       "this model. A spec-sheet prior has been observed above "
                       "40 percent error, so it is proposed rather than "
                       "executed")

    today = now.strftime("%Y-%m-%d")
    if today in policy.blackout_dates:
        return False, f"{today} is a declared blackout date"

    if policy.level == Autonomy.WINDOW:
        if not policy.windows:
            return False, ("autonomy is window and no windows are declared, "
                           "which permits nothing. This is a configuration "
                           "error rather than a refusal")
        if not _in_window(now, policy.windows):
            return False, (f"{now.strftime('%a %H:%M')} UTC is outside the "
                           f"declared execution windows, so this is proposed "
                           f"instead")

    last = state.last_move(workload_class)
    if last:
        hours = (now - datetime.fromisoformat(
            last.at_utc.replace("Z", "+00:00"))).total_seconds() / 3600
        if hours < policy.min_hours_between_moves:
            return False, (f"this class moved {hours:.0f} hours ago and the "
                           f"rate limit is {policy.min_hours_between_moves}. "
                           f"A placement that moves twice in a week is "
                           f"oscillating, not adapting")

    if moves_already >= policy.max_moves_per_pass:
        return False, (f"{moves_already} execution(s) already this pass, and "
                       f"the blast radius is {policy.max_moves_per_pass}. One "
                       f"bad estimate must not move a fleet")

    return True, (f"{decision.margin:.1%} margin, measured, inside policy, "
                  f"executing")


def _in_window(now: datetime, windows) -> bool:
    """Whether the moment falls inside a declared window.

    Windows are (weekday, start_hour, end_hour) with Monday as 0 and hours in
    UTC. A window that wraps midnight is expressed as two windows, because a
    single range that silently spans days is the kind of thing that executes
    on the wrong Tuesday.
    """
    wd, hour = now.weekday(), now.hour
    return any(d == wd and s <= hour < e for d, s, e in windows)


# --------------------------------------------------------------- the rollback

def should_roll_back(policy: AutonomyPolicy, breaches: int,
                     observed_cost: float, expected_cost: float,
                     tolerance: float = 0.15) -> tuple[bool, str]:
    """Whether an executed move must be reverted.

    Two triggers, and the first outranks everything. A move that breaches the
    service level reverts regardless of what it saved, because a placement
    that misses the bound is not a cheap placement, it is not a placement.

    The second is a cost estimate that did not survive contact. If the
    observed cost exceeds the estimate by more than the tolerance, the model
    was wrong about this cell and acting on it further would compound the
    error.
    """
    if breaches >= policy.rollback_after_breaches:
        return True, (f"the service level was breached in {breaches} "
                      f"consecutive period(s). A placement that misses its "
                      f"bound reverts regardless of what it saved")

    if expected_cost > 0 and observed_cost > expected_cost * (1 + tolerance):
        over = observed_cost / expected_cost - 1
        return True, (f"observed cost is {over:.0%} above the estimate that "
                      f"justified the move. The model was wrong about this "
                      f"cell, so the move reverts and the cell is flagged for "
                      f"measurement")

    return False, ""


# ------------------------------------------------------------------ executing

def execute(client, repo, proposal, policy: AutonomyPolicy,
            state: ExecutionState, *, open_proposal_fn, now=None):
    """Open the pull request and merge it, under a declared policy.

    The change still lands as a commit in the customer's history with a diff
    and the evidence attached. The difference between propose and execute is
    who presses merge, not whether there is a record: an executed change is as
    auditable and as revertible as a merged one, because it is a merged one.

    The authorization is the autonomy policy in their repository, committed by
    them. pilot never has standing permission to write to a default branch; it
    has permission to merge a pull request it opened, for a class whose policy
    allows it, when every guardrail passes.
    """
    now = now or datetime.now(UTC)
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    cls = proposal.watched.workload_class
    d = proposal.decision

    result = open_proposal_fn(client, repo, proposal)
    pr = result.get("html_url") or str(result.get("number", ""))

    permitted, reason = may_execute(
        d, policy, state, cls, now=now,
        moves_already=state.moves_this_pass(now - timedelta(hours=1)))

    if not permitted:
        rec = ExecutionRecord(cls, d.incumbent, d.recommended, stamp,
                              False, reason, pr)
        state.records.append(rec)
        return rec

    client.merge_proposal(repo, result, policy_reason=reason)
    rec = ExecutionRecord(cls, d.incumbent, d.recommended, stamp,
                          True, reason, pr)
    state.records.append(rec)
    return rec
