"""Silicon profiles: the hardware side of the roofline model.

Each profile carries peak specs plus *achievable-efficiency* knobs (MFU,
bandwidth efficiency). Peak specs alone are marketing numbers; the efficiency
factors are where real-world calibration lives. In a production system these
would be fitted from measured traces (this is the slot for an empirical
calibration layer on top of the analytical model).

All FLOPS are dense FP16/BF16 (no sparsity inflation).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SiliconProfile:
    name: str
    silicon_class: str          # "gpu" | "cpu" | "asic" | "fpga"
    peak_tflops: float          # dense fp16/bf16 TFLOPS
    hbm_bw_tbs: float           # memory bandwidth, TB/s
    mem_gb: float               # device memory, GB
    base_price_hr: float        # on-demand $/device-hour (reference price)
    mfu: float = 0.50           # achievable fraction of peak compute
    bw_eff: float = 0.75        # achievable fraction of peak bandwidth
    prefill_overhead_ms: float = 0.0  # fixed per-request latency (kernel
                                # launch + server scheduling); measured on
                                # real hardware, ~75-145ms on L40S/vLLM. Not
                                # compute-physics; dominates short-context TTFT.
    tp_eff: float = 0.85        # per-doubling tensor-parallel scaling efficiency

    @property
    def ridge_flops_per_byte(self) -> float:
        """Arithmetic intensity above which this silicon is compute-bound."""
        return (self.peak_tflops * 1e12) / (self.hbm_bw_tbs * 1e12)


# Reference fleet. Prices are representative on-demand rates; the SimBackend
# applies a drifting spot multiplier on top of these.
FLEET: dict[str, SiliconProfile] = {
    p.name: p
    for p in [
        SiliconProfile("h100-sxm",  "gpu", peak_tflops=989,  hbm_bw_tbs=3.35, mem_gb=80,  base_price_hr=3.20),
        SiliconProfile("h200-sxm",  "gpu", peak_tflops=989,  hbm_bw_tbs=4.80, mem_gb=141, base_price_hr=3.90),
        SiliconProfile("mi300x",    "gpu", peak_tflops=1307, hbm_bw_tbs=5.30, mem_gb=192, base_price_hr=2.80,
                       mfu=0.35, bw_eff=0.65),  # software maturity discount — this gap IS the placement premium source
        SiliconProfile("a100-80g",  "gpu", peak_tflops=312,  hbm_bw_tbs=2.00, mem_gb=80,  base_price_hr=1.40),
        SiliconProfile("l40s",      "gpu", peak_tflops=362,  hbm_bw_tbs=0.864, mem_gb=48, base_price_hr=0.95,
                       prefill_overhead_ms=100.0),  # measured P0 2026-07-07
        SiliconProfile("cpu-spr",   "cpu", peak_tflops=8,    hbm_bw_tbs=0.30, mem_gb=512, base_price_hr=0.60,
                       mfu=0.30, bw_eff=0.60),
    ]
}
