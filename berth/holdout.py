"""Holdout assignment and period logic for the Placement Holdout Protocol.

This is the reference implementation of the protocol published at
reckonresearch.com. It is deliberately small and deliberately dependency-free,
because a protocol whose hardest part is a hash function nobody wants to write
themselves does not get adopted.

Two things live here:

**Assignment.** Which leg a request runs on, decided at admission from a hash
of the request identifier and a seed committed before the period opened. The
customer runs this, inside their own admission path or at their gateway. We
never see a request.

**Period bookkeeping.** When a period opens, when warm-up ends, when it
closes, and the four stationarity checks that decide whether it can be billed.

What is not here: anything that touches traffic, anything that decides where a
workload should run, and anything that computes money. Placement is in
berth.place and settlement is in berth.receipt, separated because the party
being measured should be able to run this file without running either of the
others.
"""

from __future__ import annotations

import hashlib
import statistics as st
from dataclasses import dataclass, field


class ProtocolError(ValueError):
    """A period was configured or used in a way the protocol forbids."""


# --------------------------------------------------------------- assignment

def assign(request_id: str, seed: str, holdout_fraction: float) -> str:
    """Return "baseline" or "treatment" for one request.

    SHA-256 truncated to 64 bits, with the seed prepended rather than appended.
    Prepending matters: sequential identifiers, tenant-prefixed identifiers and
    timestamps all carry structure in their leading bytes, and a hash seeded
    only at the end can let that structure survive into the assignment. A
    tenant whose identifiers sort together would then land disproportionately
    on one leg, and the measured difference would be a property of the traffic
    rather than of the placement.

    Deterministic, so either party can recompute any request's assignment from
    the published seed. That is what makes a dispute resolvable by rerunning
    arithmetic rather than by argument.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ProtocolError(
            f"holdout_fraction must be strictly between 0 and 1, got "
            f"{holdout_fraction}. A fraction of 0 measures nothing and a "
            f"fraction of 1 changes nothing.")
    if not seed:
        raise ProtocolError(
            "an empty seed makes the assignment unreproducible, which removes "
            "the only defence against selective measurement")
    digest = hashlib.sha256(f"{seed}:{request_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return "baseline" if bucket < holdout_fraction else "treatment"


def realised_fraction(request_ids, seed: str, holdout_fraction: float) -> float:
    """Share of a set of identifiers that lands on the baseline leg.

    Run before a period opens, against a sample of real identifiers. A hash
    that does not distribute evenly over a customer's identifier format is a
    defect in the instrument rather than a result, and it is much cheaper to
    find here than in settlement.
    """
    ids = list(request_ids)
    if not ids:
        raise ProtocolError("no identifiers to check")
    n = sum(1 for r in ids if assign(r, seed, holdout_fraction) == "baseline")
    return n / len(ids)


# ------------------------------------------------------------- declaration

@dataclass(frozen=True)
class Declaration:
    """Everything agreed before the first request moves.

    Frozen because section 1 of the protocol voids a period if any of it
    changes mid-flight. Making that a language-level guarantee rather than a
    convention removes the most obvious way to accidentally invalidate a
    period.
    """

    workload_class: str
    baseline_placement: str
    treatment_placement: str
    slo_metric: str                  # e.g. "p99_ttft_ms"
    slo_bound_ms: float
    quality_floor: str               # declared by the customer, evaluated by them
    holdout_fraction: float
    seed: str
    annual_class_spend: float
    share: float = 0.20
    # Reverts immediately rather than at the period boundary, because a
    # boundary can be thirty days away and a breached service level cannot
    # wait that long.
    breaker_margin: float = 0.20
    breaker_window_minutes: int = 15
    breaker_min_requests: int = 200

    def __post_init__(self):
        if not 0.0 < self.holdout_fraction < 1.0:
            raise ProtocolError("holdout_fraction must be between 0 and 1")
        if not 0.0 < self.share < 1.0:
            raise ProtocolError("share must be between 0 and 1")
        if self.slo_bound_ms <= 0:
            raise ProtocolError("an SLO bound must be positive")
        if not self.seed:
            raise ProtocolError("the seed must be committed before the period")

    @property
    def instrument(self) -> str:
        """Flat fee or savings-share, decided by arithmetic rather than by
        negotiation. Below the break-even the holdout costs more than the
        arrangement returns."""
        s_star = scale_floor(self.share)
        if self.annual_class_spend < 250_000:
            return "assurance_light"
        if self.annual_class_spend < s_star:
            return "assurance"
        return "savings_share"


def scale_floor(share: float = 0.20, flat_fee: float = 15_000.0,
                premium: float = 0.25, holdout_fraction: float = 0.05) -> float:
    """Annual class spend above which savings-share beats a flat fee.

        S* = flat_fee / (share x premium x (1 - 2h))

    At the defaults this is $333,333. Below it the holdout, which runs on the
    worse placement by design, costs more than the arrangement returns.
    """
    denom = share * premium * (1 - 2 * holdout_fraction)
    if denom <= 0:
        raise ProtocolError("share, premium and holdout give a non-positive "
                            "denominator")
    return flat_fee / denom


# ------------------------------------------------------------------ period

@dataclass
class LegObservations:
    """What one leg produced over a window. Supplied by the customer."""

    spend: float                     # metered cost of the capacity, including idle
    requests: int
    compliant_requests: int          # met the bound AND the quality floor
    served_tokens: int               # output tokens from compliant requests only
    # For the stationarity checks: per-half aggregates and the context
    # distribution. Kept as summaries rather than raw traces so a customer can
    # supply them without shipping their traffic anywhere.
    first_half_requests: int = 0
    second_half_requests: int = 0
    first_half_cost_per_mtok: float = 0.0
    second_half_cost_per_mtok: float = 0.0
    median_context: float = 0.0
    median_generation: float = 0.0
    context_deciles: list[float] = field(default_factory=list)

    @property
    def compliance_rate(self) -> float:
        return self.compliant_requests / self.requests if self.requests else 0.0

    @property
    def cost_per_mtok(self) -> float:
        """Cost per million served tokens.

        Denominator is compliant output only. A token delivered after its
        deadline is not a cheap token, it is not a token, and a response
        failing the quality floor is work that was paid for and cannot be
        sold.
        """
        if self.served_tokens <= 0:
            return float("inf")
        return self.spend / (self.served_tokens / 1e6)


def wilson_interval(successes: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion.

    Wilson rather than normal because compliance legitimately sits near zero
    and near one, where the normal approximation produces bounds outside the
    unit interval and quietly stops meaning anything.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def decile_divergence(a, b) -> float:
    """Largest relative gap between two distributions' decile boundaries.

    A median can hold while the distribution moves underneath it. A class
    serving two tenants at 500 and 8,000 tokens has the same median whether
    the split is 50/50 or 20/80, and the cost per served token is very
    different.

    Deciles rather than a Kolmogorov-Smirnov test, because the customer
    supplies summaries rather than raw traffic. KS applied to ten boundary
    values treats them as a ten-point sample, and its statistic then moves in
    steps of 0.1 regardless of how close the distributions are, which puts
    every comparison on a threshold boundary. Comparing the boundaries
    directly is both the right object and interpretable: a 4 percent figure
    means the two legs' context lengths differ by at most 4 percent at every
    decile.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    worst = 0.0
    for x, y in zip(sorted(a), sorted(b), strict=False):
        m = (x + y) / 2
        if m > 0:
            worst = max(worst, abs(x - y) / m)
    return worst


def stationarity(baseline: LegObservations, treatment: LegObservations,
                 tolerance: float = 0.10, mix_bound: float = 0.10):
    """The four checks that decide whether a period can be billed.

    Returns (verdicts, blocking). A period failing 4.1 through 4.3 is
    reported and not billed; 4.3b flags for a decision rather than voiding,
    because a mix shift explains a difference without necessarily
    invalidating it.
    """
    v, blocking = {}, False

    def diverge(x, y):
        m = (x + y) / 2
        return abs(x - y) / m if m else 0.0

    load = max(diverge(baseline.first_half_requests, baseline.second_half_requests),
               diverge(treatment.first_half_requests, treatment.second_half_requests))
    v["4.1_offered_load"] = (load, load <= tolerance)

    cost = max(diverge(baseline.first_half_cost_per_mtok,
                       baseline.second_half_cost_per_mtok),
               diverge(treatment.first_half_cost_per_mtok,
                       treatment.second_half_cost_per_mtok))
    v["4.2_cost_per_mtok"] = (cost, cost <= tolerance)

    shape = max(diverge(baseline.median_context, treatment.median_context),
                diverge(baseline.median_generation, treatment.median_generation))
    v["4.3_shape_medians"] = (shape, shape <= tolerance)

    mix = decile_divergence(baseline.context_deciles, treatment.context_deciles)
    v["4.3b_context_mix"] = (mix, mix <= mix_bound)

    for key in ("4.1_offered_load", "4.2_cost_per_mtok", "4.3_shape_medians"):
        if not v[key][1]:
            blocking = True
    return v, blocking


def warmup_complete(cost_per_mtok_intervals, tolerance: float = 0.05,
                    min_intervals: int = 2) -> bool:
    """Whether a newly started placement has stabilised.

    A cold placement has empty caches, unloaded weights and uncaptured
    compiled graphs. Comparing a warm baseline against a cold treatment
    measures the start rather than the placement, and the bias runs against
    whichever leg was restarted.
    """
    if len(cost_per_mtok_intervals) < min_intervals:
        return False
    recent = cost_per_mtok_intervals[-min_intervals:]
    if min(recent) <= 0:
        return False
    return (max(recent) - min(recent)) / st.mean(recent) <= tolerance


def breaker_tripped(declaration: Declaration, treatment_p99_ms: float,
                    window_requests: int) -> bool:
    """Whether the treatment leg has breached badly enough to revert now.

    A period boundary can be thirty days away. A placement that breaks the
    service level needs reverting in minutes, and a period ended this way is
    voided rather than counted as a loss, because a placement that never held
    the bound was never a measurement.
    """
    if window_requests < declaration.breaker_min_requests:
        return False
    return treatment_p99_ms > declaration.slo_bound_ms * (1 + declaration.breaker_margin)
