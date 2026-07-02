"""Tail-latency model: M/M/c queueing on batch slots.

Model: a replica exposes `batch` slots (continuous batching admission). A
request occupies one slot for its full residence time D = TTFT + out_tokens
x TPOT. Requests arrive Poisson(lambda). The fleet of R replicas is an M/M/c
queue with c = R x batch servers and per-slot service rate mu = 1/D.

Why M/M/c and not simulation: closed form, numerically stable, auditable —
every p99 is derivable by hand from (lambda, mu, c). The known bias: real
residence times are not exponential (output-length distributions are
heavy-tailed), so true tails are somewhat worse than modeled. The standard
correction is an SCV inflation factor (Allen-Cunneen); it belongs in the
calibration layer once measured residence-time variance exists.

Formulas:
  Erlang B recurrence (stable): B(0)=1; B(k) = a*B(k-1) / (k + a*B(k-1))
  Erlang C:  C = B / (1 - rho * (1 - B)),  rho = a/c,  a = lambda/mu
  M/M/c wait tail is exponential: P(W > t) = C * exp(-(c*mu - lambda) * t)
  => p99 wait = ln(C / 0.01) / (c*mu - lambda)   (0 if C <= 0.01)
"""

import math
from dataclasses import dataclass

MAX_REPLICAS = 64
DEFAULT_TARGET_UTIL = 0.85   # sizing convention when no TTFT SLO is given —
                             # a convention, not a fundamental constraint


def erlang_c(c: int, offered_load: float) -> float:
    """P(request waits) for M/M/c. Requires offered_load < c (else returns 1.0)."""
    if offered_load >= c:
        return 1.0
    b = 1.0
    for k in range(1, c + 1):
        b = offered_load * b / (k + offered_load * b)   # Erlang B recurrence
    rho = offered_load / c
    return b / (1.0 - rho * (1.0 - b))


def p99_wait_s(c: int, mu: float, lam: float) -> float:
    """99th percentile queueing delay in seconds for M/M/c."""
    a = lam / mu
    pc = erlang_c(c, a)
    if pc <= 0.01:
        return 0.0
    return math.log(pc / 0.01) / (c * mu - lam)


@dataclass(frozen=True)
class FleetSizing:
    feasible: bool
    replicas: int = 0
    utilization: float = 0.0
    p99_ttft_ms: float = float("inf")
    reason: str = ""


def size_replicas(arrival_rps: float, residence_s: float, batch: int,
                  ttft_mean_ms: float, p99_ttft_slo_ms: float | None) -> FleetSizing:
    """Minimum replicas meeting the p99 TTFT SLO (or the utilization target).

    p99 TTFT = p99 queueing wait + mean prefill service time. Prefill service
    variance is small relative to queueing delay at meaningful load, so the
    wait term dominates the tail; this is the honest first-order model.
    """
    mu = 1.0 / residence_s
    offered = arrival_rps * residence_s        # load in slot-equivalents

    r = max(1, math.ceil(offered / batch / 0.98))   # start just under saturation
    while r <= MAX_REPLICAS:
        c = r * batch
        util = offered / c
        if util < 1.0:
            wait_ms = p99_wait_s(c, mu, arrival_rps) * 1e3
            p99_ttft = wait_ms + ttft_mean_ms
            if p99_ttft_slo_ms is not None:
                if p99_ttft <= p99_ttft_slo_ms:
                    return FleetSizing(True, r, util, p99_ttft)
            elif util <= DEFAULT_TARGET_UTIL:
                return FleetSizing(True, r, util, p99_ttft)
        r += 1

    return FleetSizing(False, reason=f"needs >{MAX_REPLICAS} replicas for SLO at {arrival_rps} rps")
