"""Measured traces: the schema calibration consumes.

A TraceRecord is one observation of real serving behavior: what ran, where,
and the measured latencies. In production these come from the serving layer's
metrics pipeline. Here, `generate_traces` produces them from a *hidden* fleet
(true efficiencies unknown to the calibrator) plus multiplicative lognormal
measurement noise — so tests can prove blind parameter recovery, which is the
only honest validation available without hardware.
"""

import math
import random
from dataclasses import dataclass, replace

from .estimate import estimate
from .silicon import SiliconProfile
from .workload import MODELS, ModelSpec, WorkloadSpec, profile


@dataclass(frozen=True)
class TraceRecord:
    silicon: str
    model_name: str
    batch: int
    avg_prompt_tokens: int
    avg_output_tokens: int
    measured_ttft_ms: float
    measured_tpot_ms: float
    t: float = 0.0                  # normalized observation time in [0, 1]

    def signature(self, models: dict[str, ModelSpec] = MODELS):
        return profile(WorkloadSpec(
            model=models[self.model_name],
            avg_prompt_tokens=self.avg_prompt_tokens,
            avg_output_tokens=self.avg_output_tokens,
            target_batch=self.batch,
        ))


def make_true_fleet(prior_fleet: dict[str, SiliconProfile],
                    true_efficiencies: dict[str, tuple[float, float]]) -> dict[str, SiliconProfile]:
    """Fleet with ground-truth (mfu, bw_eff) substituted — hidden from the fitter."""
    return {
        name: replace(hw, mfu=true_efficiencies[name][0], bw_eff=true_efficiencies[name][1])
        if name in true_efficiencies else hw
        for name, hw in prior_fleet.items()
    }


def generate_traces(true_fleet: dict[str, SiliconProfile],
                    n_per_silicon: int = 40,
                    noise_sigma: float = 0.05,
                    seed: int = 0,
                    models: dict[str, ModelSpec] = MODELS,
                    eff_fn=None) -> list[TraceRecord]:
    """Sample workload mixes, run the TRUE model, add measurement noise.

    eff_fn(silicon, sig, t) -> (mfu, bw_eff), optional: overrides the true
    fleet's efficiencies per observation. One hook covers constant truth
    (None), per-workload-class truth (inspect sig), and software-stack drift
    (inspect t). Ground-truth generation and hypothesis structure stay
    decoupled from the fitter under test.
    """
    rng = random.Random(seed)
    traces: list[TraceRecord] = []
    model_names = list(models)
    for name, hw in true_fleet.items():
        for i in range(n_per_silicon):
            t = i / (n_per_silicon - 1) if n_per_silicon > 1 else 0.0
            mn = rng.choice(model_names)
            w = WorkloadSpec(
                model=models[mn],
                avg_prompt_tokens=rng.choice([256, 512, 1024, 2048, 4096]),
                avg_output_tokens=rng.choice([64, 128, 256, 512]),
                target_batch=rng.choice([1, 4, 8, 16, 32]),
            )
            sig = profile(w)
            hw_i = hw
            if eff_fn is not None:
                mfu_i, bw_i = eff_fn(name, sig, t)
                hw_i = replace(hw, mfu=mfu_i, bw_eff=bw_i)
            e = estimate(sig, hw_i, hw.base_price_hr)
            if not e.feasible:
                continue
            noise = lambda: math.exp(rng.gauss(0.0, noise_sigma))
            traces.append(TraceRecord(
                silicon=name, model_name=mn, batch=w.target_batch,
                avg_prompt_tokens=w.avg_prompt_tokens,
                avg_output_tokens=w.avg_output_tokens,
                measured_ttft_ms=e.ttft_ms * noise(),
                measured_tpot_ms=e.tpot_ms * noise(),
                t=t,
            ))
    return traces
