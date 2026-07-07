"""The `estimate` primitive: signature x silicon -> predicted performance/cost.

Pure analytical roofline. Deliberately transparent: every number is derivable
from public specs plus two efficiency factors per device. A neutral reference
standard must be auditable; a fitted black-box model can layer on top for
calibration, but the schema is set here.

Model:
  decode step time = max(compute_time, memory_time) / (nothing hidden)
    compute_time = batch * flops_per_token / (n_dev * peak * mfu * tp_scale)
    memory_time  = (weights + batch * context * kv) / (n_dev * bw * bw_eff * tp_scale)
  prefill time   = prefill_flops / effective_compute   (compute-bound regime)

Tensor parallelism: n_dev chosen as the minimum that fits weights + KV in
0.9 * aggregate memory. Scaling efficiency decays per doubling (tp_eff^log2).
"""

import math
from dataclasses import dataclass

from .silicon import SiliconProfile
from .workload import ComputeSignature

MEM_HEADROOM = 0.90  # fraction of device memory usable for weights + KV


@dataclass(frozen=True)
class Estimate:
    silicon: str
    feasible: bool
    reason: str = ""
    n_devices: int = 0
    ttft_ms: float = float("inf")
    tpot_ms: float = float("inf")
    tokens_per_s: float = 0.0           # aggregate output tokens/s per replica
    replica_price_hr: float = 0.0       # n_devices * current $/hr
    cost_per_mtok: float = float("inf") # $ per 1M output tokens
    bound: str = ""                     # "compute" | "memory" — which roof binds
    placement_premium: float = 0.0           # filled in by client: cost/best_cost - 1
    # Fleet-sizing fields (populated only when the workload carries arrival_rps):
    replicas: int = 1
    utilization: float = 0.0            # offered load / total slots
    p99_ttft_ms: float | None = None    # queueing wait p99 + prefill service


def replica_layout(sig: ComputeSignature, hw: SiliconProfile) -> tuple[int, float]:
    """(n_devices, tp_scale) for this signature on this silicon.

    Shared by estimator and calibrator — the trace-inversion math must use the
    exact same layout formulas or fitted parameters absorb layout error.
    """
    kv_total = sig.batch * sig.avg_context * sig.model.kv_bytes_per_token
    need_bytes = sig.model.weight_bytes + kv_total
    n_dev = max(1, math.ceil(need_bytes / (hw.mem_gb * 1e9 * MEM_HEADROOM)))
    tp_scale = hw.tp_eff ** math.log2(n_dev) if n_dev > 1 else 1.0
    return n_dev, tp_scale


def estimate(sig: ComputeSignature, hw: SiliconProfile, price_hr: float) -> Estimate:
    m = sig.model
    kv_total = sig.batch * sig.avg_context * m.kv_bytes_per_token

    n_dev, tp_scale = replica_layout(sig, hw)
    if n_dev > 8:
        return Estimate(hw.name, feasible=False, reason=f"needs {n_dev} devices (>8/node)")
    eff_flops = n_dev * hw.peak_tflops * 1e12 * hw.mfu * tp_scale
    eff_bw = n_dev * hw.hbm_bw_tbs * 1e12 * hw.bw_eff * tp_scale

    # --- Decode (steady state) ---
    active_weight_bytes = m.active_params_b * 1e9 * m.bytes_per_param
    compute_t = sig.batch * sig.decode_flops_per_token / eff_flops
    memory_t = (active_weight_bytes + kv_total) / eff_bw
    step_t = max(compute_t, memory_t)
    bound = "compute" if compute_t >= memory_t else "memory"

    tpot_ms = step_t * 1e3
    tokens_per_s = sig.batch / step_t

    # --- Prefill ---
    # Compute-bound service time PLUS a fixed per-request overhead (kernel
    # launch + server scheduling). P0 on real L40S showed ~100ms of fixed
    # latency that dominates short-context TTFT; omitting it made implied
    # MFU spuriously trend with context length. Prefill MAPE 28%->11%.
    ttft_ms = (sig.prefill_flops_per_req / eff_flops) * 1e3 + hw.prefill_overhead_ms

    # --- SLO feasibility ---
    if sig.p99_ttft_ms is not None and ttft_ms > sig.p99_ttft_ms:
        return Estimate(hw.name, feasible=False, n_devices=n_dev, ttft_ms=ttft_ms,
                        tpot_ms=tpot_ms,
                        reason=f"TTFT {ttft_ms:.0f}ms > SLO {sig.p99_ttft_ms:.0f}ms")
    if sig.p99_tpot_ms is not None and tpot_ms > sig.p99_tpot_ms:
        return Estimate(hw.name, feasible=False, n_devices=n_dev, ttft_ms=ttft_ms,
                        tpot_ms=tpot_ms,
                        reason=f"TPOT {tpot_ms:.1f}ms > SLO {sig.p99_tpot_ms:.1f}ms")

    replica_price = n_dev * price_hr

    # --- Fleet sizing (tail-aware path) ---
    if sig.arrival_rps is not None:
        from .queueing import size_replicas
        residence_s = (ttft_ms + sig.avg_output_tokens * tpot_ms) / 1e3
        sizing = size_replicas(sig.arrival_rps, residence_s, sig.batch,
                               ttft_ms, sig.p99_ttft_ms)
        if not sizing.feasible:
            return Estimate(hw.name, feasible=False, n_devices=n_dev,
                            ttft_ms=ttft_ms, tpot_ms=tpot_ms, reason=sizing.reason)
        # SLO on TTFT now judged at the p99 (queueing wait + service), which
        # size_replicas already enforced. Cost includes the headroom: total
        # fleet $/hr over the goodput actually delivered at arrival_rps.
        goodput_tok_s = sig.arrival_rps * sig.avg_output_tokens
        fleet_price = sizing.replicas * replica_price
        cost_per_mtok = (fleet_price / (goodput_tok_s * 3600.0)) * 1e6
        return Estimate(
            silicon=hw.name, feasible=True, n_devices=n_dev,
            ttft_ms=ttft_ms, tpot_ms=tpot_ms, tokens_per_s=tokens_per_s,
            replica_price_hr=fleet_price, cost_per_mtok=cost_per_mtok, bound=bound,
            replicas=sizing.replicas, utilization=sizing.utilization,
            p99_ttft_ms=sizing.p99_ttft_ms,
        )

    # --- Single-replica path (no arrival rate given): v0.2 semantics ---
    cost_per_mtok = (replica_price / (tokens_per_s * 3600.0)) * 1e6

    return Estimate(
        silicon=hw.name, feasible=True, n_devices=n_dev,
        ttft_ms=ttft_ms, tpot_ms=tpot_ms, tokens_per_s=tokens_per_s,
        replica_price_hr=replica_price, cost_per_mtok=cost_per_mtok, bound=bound,
    )
