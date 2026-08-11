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

SOURCES = ("measured", "mock")
SILICON_PROVENANCE = ("captured", "self_reported", "mock")


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
    # Quant of THIS measured cell. Not cosmetic: a fp8 cell must be inverted
    # against a fp8 signature (weights/KV bytes + dtype peak), or the fitted
    # mfu/bw_eff silently absorb the dtype delta and corrupt the premium.
    # Default 2.0/2.0 = bf16, so pre-quant traces load unchanged.
    w_bytes: float = 2.0
    kv_bytes: float = 2.0
    source: str = "measured"        # "measured" | "mock". Schema 2.
    # How the silicon identity in this record was established. Schema 3.
    #   captured      nvidia-smi on the box agreed with --silicon
    #   self_reported the operator asserted it and nothing verified it
    #   mock          no hardware involved
    # The corpus guarantee is that every cell was observed on the hardware it
    # names. `source` proves hardware was involved at all; this proves the
    # hardware was the hardware claimed. Without it a trace from an RTX 6000
    # labelled h100-pcie passes every check, because nothing in it is a lie
    # except the label, and the label is the part the corpus indexes on.
    silicon_provenance: str = "self_reported"

    # Device power during the sweep, watts, median across samples. Schema 5.
    #
    # Added now because the schema can still be versioned cheaply. There are no
    # external contributors yet, so a field costs a version bump; once there
    # are, it costs a migration. The window closes on its own.
    #
    # Energy is a declared dimension of the unit and has never been a measured
    # term in this instrument. Stating it that way rather than claiming the
    # instrument covers it is the same rule that governs every other number
    # here: the unit names energy, the meter does not yet weigh it, and the
    # gap belongs in the open rather than in a footnote.
    #
    # Device boundary only. No facility multiplier, no PUE, no assumed
    # cooling overhead. Those are real and they are not measurable from a
    # rented instance, and multiplying a measured figure by an assumed one
    # produces a number whose provenance cannot be stated.
    power_w: float | None = None
    # How the figure was obtained: "sampled" from nvidia-smi or rocm-smi
    # during the run, or absent where the platform exposes nothing. A TPU or
    # a Trainium reports no per-device power to a tenant, so those cells will
    # carry None and that is the honest record rather than a gap to fill.
    power_provenance: str | None = None

    def __post_init__(self):
        # A mock trace and a hardware trace are otherwise indistinguishable on
        # disk, and the contribution path is a pull request. One unlabelled
        # simulated file would silently corrupt the corpus, which is the single
        # error this project cannot recover from. So the field is mandatory in
        # substance even though it has a default, and an unknown value is fatal
        # rather than coerced.
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        if self.silicon_provenance not in SILICON_PROVENANCE:
            raise ValueError(f"silicon_provenance must be one of "
                             f"{SILICON_PROVENANCE}, got {self.silicon_provenance!r}")

    def signature(self, models: dict[str, ModelSpec] = MODELS):
        model = models[self.model_name]
        if (self.w_bytes, self.kv_bytes) != (model.bytes_per_param, model.bytes_per_kv):
            model = model.quantized(self.w_bytes, self.kv_bytes)
        return profile(WorkloadSpec(
            model=model,
            avg_prompt_tokens=self.avg_prompt_tokens,
            avg_output_tokens=self.avg_output_tokens,
            target_batch=self.batch,
        ))


def make_true_fleet(prior_fleet: dict[str, SiliconProfile],
                    true_efficiencies: dict[str, tuple[float, float]]) -> dict[str, SiliconProfile]:
    """Fleet with ground-truth (mfu, bw_eff) substituted, hidden from the fitter.

    Any measured ladder is stripped. A synthetic ground truth is defined
    entirely by the pair being recovered, and leaving a ladder in the
    generator would put a third parameter in it that the fitter cannot see and
    cannot recover, so recovery would fail for a reason unrelated to the
    estimator.
    """
    return {
        name: replace(hw, mfu=true_efficiencies[name][0],
                      bw_eff=true_efficiencies[name][1], kv_bw_ladder=None)
        if name in true_efficiencies else replace(hw, kv_bw_ladder=None)
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
                source="mock",   # generated from a hidden fleet, not observed
            ))
    return traces
