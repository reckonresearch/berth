"""Calibration: fit per-silicon (mfu, bw_eff) from measured traces.

Method: direct inversion, not generic optimization. The roofline separates:

  TTFT (prefill, compute-bound by construction):
      mfu = prefill_flops / (ttft * n_dev * peak * tp_scale)
  TPOT, memory-bound decode:
      bw_eff = step_bytes / (tpot * n_dev * bw * tp_scale)
  TPOT, compute-bound decode:
      mfu = batch * flops_per_token / (tpot * n_dev * peak * tp_scale)

Every trace yields closed-form parameter estimates; aggregate with medians
(robust to noise and the occasional outlier trace). Which roof binds depends
on the parameters being fitted, so iterate fit -> reclassify -> refit; this
fixed point converges in 2 passes for smooth efficiency landscapes.

Every fitted value is traceable to specific observations — auditable by
construction, which is the property a neutral reference model must keep.
"""

import random
from dataclasses import dataclass, field, replace
from statistics import mean, median

from .estimate import replica_layout
from .silicon import SiliconProfile
from .traces import TraceRecord

_EFF_MIN, _EFF_MAX = 0.05, 1.0  # physical bounds on any efficiency factor


@dataclass(frozen=True)
class CalibrationReport:
    fitted: dict[str, tuple[float, float]]          # silicon -> (mfu, bw_eff)
    n_traces: dict[str, int]
    mape_prior: float                                # holdout MAPE before
    mape_calibrated: float                           # holdout MAPE after
    # 95% bootstrap CIs: silicon -> ((mfu_lo, mfu_hi), (bw_lo, bw_hi)).
    # A point fit without an interval is an opinion; the index needs error
    # bars, and downstream placement can widen hysteresis when CIs are wide.
    ci95: dict[str, tuple[tuple[float, float], tuple[float, float]]] = field(
        default_factory=dict)


def _clamp(x: float) -> float:
    return min(_EFF_MAX, max(_EFF_MIN, x))


def _fit_one(hw: SiliconProfile, traces: list[TraceRecord], n_iters: int = 2) -> SiliconProfile:
    fitted = hw
    for _ in range(n_iters):
        mfu_obs: list[float] = []
        bw_obs: list[float] = []
        for t in traces:
            sig = t.signature()
            n_dev, tp_scale = replica_layout(sig, hw)
            if n_dev > 8:
                continue
            m = sig.model

            # Prefill inversion -> mfu. Restricted to batch == 1, mirroring
            # bench.validate.check_prefill. prefill_flops_per_req is a SINGLE
            # request's work, but P0 measured that a synchronous batch is
            # prefilled SERIALLY (c_eff ~ 1.0 on L40S and H100 PCIe). Pooling
            # batches therefore attributes `batch` sequential prefills to one
            # request's compute and drags the fit toward zero (measured: 0.064
            # pooled vs 0.44 batch-1 on the same traces), which then poisons
            # every compute-bound TPOT prediction downstream. Batch-1 cells are
            # contention-free and are the only valid inversion.
            if sig.batch == 1:
                ttft_s = t.measured_ttft_ms / 1e3
                # Subtract the fixed prefill overhead before inverting, else
                # short-context TTFT (overhead-dominated) yields spuriously low
                # MFU that trends up with context. (P0 finding.)
                ttft_compute_s = max(1e-6, ttft_s - hw.prefill_overhead_ms / 1e3)
                mfu_obs.append(_clamp(
                    sig.prefill_flops_per_req / (ttft_compute_s * n_dev * hw.peak_tflops * 1e12 * tp_scale)
                ))

            # Decode inversion -> classify bound under CURRENT fit, then invert.
            tpot_s = t.measured_tpot_ms / 1e3
            step_bytes = (m.active_params_b * 1e9 * m.bytes_per_param
                          + sig.batch * sig.avg_context * m.kv_bytes_per_token)
            step_flops = sig.batch * sig.decode_flops_per_token
            compute_t = step_flops / (n_dev * hw.peak_tflops * 1e12 * fitted.mfu * tp_scale)
            memory_t = step_bytes / (n_dev * hw.hbm_bw_tbs * 1e12 * fitted.bw_eff * tp_scale)
            if memory_t >= compute_t:
                bw_obs.append(_clamp(
                    step_bytes / (tpot_s * n_dev * hw.hbm_bw_tbs * 1e12 * tp_scale)
                ))
            else:
                mfu_obs.append(_clamp(
                    step_flops / (tpot_s * n_dev * hw.peak_tflops * 1e12 * tp_scale)
                ))

        fitted = replace(
            hw,
            mfu=median(mfu_obs) if mfu_obs else hw.mfu,
            bw_eff=median(bw_obs) if bw_obs else hw.bw_eff,  # keep prior if unobserved
        )
    return fitted


def _mape(fleet: dict[str, SiliconProfile], traces: list[TraceRecord]) -> float:
    """Mean absolute % error on TTFT and TPOT predictions over traces.

    WARNING, and it is emitted at runtime: `estimate().ttft_ms` is a SINGLE
    REQUEST's prefill service time by deliberate layer separation (base =
    service time; the queueing path adds wait). Harnesses that submit a batch
    concurrently record the batch TAIL. P0 measured that no request emits a
    first token until EVERY prompt in the batch has been prefilled, so a
    batched trace's TTFT is `floor + batch * single_prefill`, not
    `floor + single_prefill`.

    Scoring one against the other compares incommensurable quantities and
    reports a large, misleading error. On the P0 traces it read 31.9% and 30.0%
    against a 15% gate; scored against the batch-tail quantity actually
    measured, the same model reads 4.4% and 4.7%. Callers with batch > 1 traces
    should score against `queueing.concurrent_prefill_ttft_ms(
    ttft_ms - prefill_overhead_ms, prefill_overhead_ms, batch)`.
    """
    from .estimate import estimate
    batched = {t.batch for t in traces if t.batch > 1}
    if batched:
        import warnings
        warnings.warn(
            f"_mape is scoring single-request TTFT against traces containing "
            f"batch sizes {sorted(batched)}. Batched traces record the batch "
            f"TAIL, so the reported TTFT error is inflated and not "
            f"interpretable as model accuracy. See _mape.__doc__.",
            RuntimeWarning, stacklevel=2)
    errs: list[float] = []
    for t in traces:
        hw = fleet[t.silicon]
        e = estimate(t.signature(), hw, hw.base_price_hr)
        if not e.feasible:
            continue
        errs.append(abs(e.ttft_ms - t.measured_ttft_ms) / t.measured_ttft_ms)
        errs.append(abs(e.tpot_ms - t.measured_tpot_ms) / t.measured_tpot_ms)
    return mean(errs) if errs else float("nan")


def calibrate(prior_fleet: dict[str, SiliconProfile],
              traces: list[TraceRecord],
              holdout_frac: float = 0.25) -> tuple[dict[str, SiliconProfile], CalibrationReport]:
    """Fit on a train split, report MAPE on a deterministic holdout split."""
    # Deterministic split: every k-th trace to holdout (no RNG -> reproducible).
    k = max(2, round(1.0 / holdout_frac))
    train = [t for i, t in enumerate(traces) if i % k != 0]
    holdout = [t for i, t in enumerate(traces) if i % k == 0]

    by_silicon: dict[str, list[TraceRecord]] = {}
    for t in train:
        by_silicon.setdefault(t.silicon, []).append(t)

    calibrated = {
        name: _fit_one(hw, by_silicon.get(name, []))
        for name, hw in prior_fleet.items()
    }
    ci95 = {
        name: _bootstrap_ci(hw, by_silicon[name])
        for name, hw in prior_fleet.items() if name in by_silicon
    }
    report = CalibrationReport(
        fitted={n: (hw.mfu, hw.bw_eff) for n, hw in calibrated.items()},
        n_traces={n: len(by_silicon.get(n, [])) for n in prior_fleet},
        mape_prior=_mape(prior_fleet, holdout),
        mape_calibrated=_mape(calibrated, holdout),
        ci95=ci95,
    )
    return calibrated, report


def _bootstrap_ci(hw: SiliconProfile, traces: list[TraceRecord],
                  n_boot: int = 200, seed: int = 0):
    """Percentile bootstrap over traces. Seeded -> reproducible intervals."""
    rng = random.Random(seed)
    mfus, bws = [], []
    for _ in range(n_boot):
        sample = [traces[rng.randrange(len(traces))] for _ in range(len(traces))]
        f = _fit_one(hw, sample)
        mfus.append(f.mfu)
        bws.append(f.bw_eff)
    mfus.sort()
    bws.sort()
    lo, hi = int(0.025 * n_boot), int(0.975 * n_boot) - 1
    return (mfus[lo], mfus[hi]), (bws[lo], bws[hi])
