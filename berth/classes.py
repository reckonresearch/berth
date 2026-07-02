"""Per-workload-class calibration and software-stack drift detection.

Why classes: a single (mfu, bw_eff) per silicon averages over genuinely
different kernel regimes — batch-1 decode is a bandwidth-starved GEMV, batch-32
is fat GEMMs; short-context attention and long-context attention hit different
code paths. Discrete buckets are used instead of a fitted efficiency surface
because each cell stays auditable: a fit, a trace count, nothing extrapolated
invisibly.

Fallback hierarchy — never fabricate from thin data:
    per-(silicon, class) fit   if the cell has >= min_traces observations
    -> per-silicon global fit  if any traces exist for the silicon
    -> prior                   otherwise

Why drift matters: efficiency factors are properties of the *software stack*,
not the die. A maturing stack (kernel releases, compiler updates) shows up as
a trending bw_eff — the placement premium moving in real time. Detection is
windowed refits + OLS slope: reuses the audited inversion fitter unchanged,
and the slope has direct units (efficiency change over the observation span).
"""

from dataclasses import dataclass, replace

from .calibrate import _fit_one
from .silicon import SiliconProfile
from .traces import TraceRecord
from .workload import ComputeSignature

# ---------------------------------------------------------------- classes ---

def workload_class(sig: ComputeSignature) -> str:
    """Bucket key: batch regime x context regime.

    Boundaries are conventions chosen to separate kernel regimes (GEMV vs
    GEMM; short vs long attention), not fundamental constraints — revisit
    against per-cell fit variance once real traces exist.
    """
    b = sig.batch
    bb = "b1-4" if b <= 4 else "b8-16" if b <= 16 else "b32+"
    c = sig.avg_context
    cb = "ctxS" if c < 1024 else "ctxM" if c <= 4096 else "ctxL"
    return f"{bb}/{cb}"


# --------------------------------------------- per-class calibrated fleet ---

@dataclass(frozen=True)
class ClassedFleet:
    prior: dict[str, SiliconProfile]
    global_fits: dict[str, SiliconProfile]                  # from calibrate()
    class_fits: dict[tuple[str, str], tuple[float, float]]  # (silicon, class) -> (mfu, bw_eff)
    class_counts: dict[tuple[str, str], int]
    min_traces: int = 12

    def profile_for(self, silicon: str, sig: ComputeSignature) -> SiliconProfile:
        key = (silicon, workload_class(sig))
        if self.class_counts.get(key, 0) >= self.min_traces:
            mfu, bw = self.class_fits[key]
            return replace(self.prior[silicon], mfu=mfu, bw_eff=bw)
        if silicon in self.global_fits:
            return self.global_fits[silicon]
        return self.prior[silicon]

    def resolver(self):
        """Adapter for PlacementClient(profile_resolver=...)."""
        return self.profile_for


def calibrate_classed(prior_fleet: dict[str, SiliconProfile],
                      traces: list[TraceRecord],
                      min_traces: int = 12) -> ClassedFleet:
    by_cell: dict[tuple[str, str], list[TraceRecord]] = {}
    by_silicon: dict[str, list[TraceRecord]] = {}
    for tr in traces:
        cell = (tr.silicon, workload_class(tr.signature()))
        by_cell.setdefault(cell, []).append(tr)
        by_silicon.setdefault(tr.silicon, []).append(tr)

    global_fits = {
        name: _fit_one(hw, by_silicon[name])
        for name, hw in prior_fleet.items() if name in by_silicon
    }
    class_fits: dict[tuple[str, str], tuple[float, float]] = {}
    for (silicon, cls), cell_traces in by_cell.items():
        fitted = _fit_one(prior_fleet[silicon], cell_traces)
        class_fits[(silicon, cls)] = (fitted.mfu, fitted.bw_eff)

    return ClassedFleet(
        prior=prior_fleet,
        global_fits=global_fits,
        class_fits=class_fits,
        class_counts={cell: len(ts) for cell, ts in by_cell.items()},
        min_traces=min_traces,
    )


# ------------------------------------------------------------------ drift ---

@dataclass(frozen=True)
class DriftSignal:
    silicon: str
    param: str              # "mfu" | "bw_eff"
    start: float            # fitted value in first window
    end: float              # fitted value in last window
    slope: float            # OLS change over the full observation span [0, 1]
    flagged: bool


def _ols_slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    var = sum((x - mx) ** 2 for x in xs)
    if var == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / var


def detect_drift(prior_fleet: dict[str, SiliconProfile],
                 traces: list[TraceRecord],
                 n_windows: int = 5,
                 threshold: float = 0.05,
                 min_per_window: int = 8) -> list[DriftSignal]:
    """Windowed refits per silicon, OLS slope per parameter over t in [0, 1].

    `threshold` is absolute efficiency change over the span; 0.05 sits well
    above median-fit jitter at 5% measurement noise (a convention — tighten it
    once real trace noise is characterized).
    """
    signals: list[DriftSignal] = []
    by_silicon: dict[str, list[TraceRecord]] = {}
    for tr in traces:
        by_silicon.setdefault(tr.silicon, []).append(tr)

    for name, ts in by_silicon.items():
        ts = sorted(ts, key=lambda tr: tr.t)
        size = len(ts) // n_windows
        if size < min_per_window:
            continue  # not enough data for a defensible trend — say nothing
        mids, mfus, bws = [], [], []
        for w in range(n_windows):
            chunk = ts[w * size:(w + 1) * size] if w < n_windows - 1 else ts[w * size:]
            fitted = _fit_one(prior_fleet[name], chunk)
            mids.append(sum(tr.t for tr in chunk) / len(chunk))
            mfus.append(fitted.mfu)
            bws.append(fitted.bw_eff)
        for param, ys in (("mfu", mfus), ("bw_eff", bws)):
            slope = _ols_slope(mids, ys)
            signals.append(DriftSignal(
                silicon=name, param=param, start=ys[0], end=ys[-1],
                slope=slope, flagged=abs(slope) > threshold,
            ))
    return signals
