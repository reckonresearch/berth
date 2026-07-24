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
        # H100 PCIe: lower clocks/power (310W) and HBM2e vs SXM's HBM3, so both
        # peak compute and bandwidth are materially below the SXM part. Measured
        # P0 microbench on this card: GEMM 388 TFLOPS bf16, d2d bandwidth
        # 1591 GB/s. prefill_overhead_ms fitted from batch=1 traces = 54.6ms.
        SiliconProfile("h100-pcie", "gpu", peak_tflops=756,  hbm_bw_tbs=2.00, mem_gb=80,  base_price_hr=2.39),
        SiliconProfile("h200-sxm",  "gpu", peak_tflops=989,  hbm_bw_tbs=4.80, mem_gb=141, base_price_hr=3.90),
        SiliconProfile("mi300x",    "gpu", peak_tflops=1307, hbm_bw_tbs=5.30, mem_gb=192, base_price_hr=2.80,
                       mfu=0.35, bw_eff=0.65),  # software maturity discount — this gap IS the placement premium source
        SiliconProfile("a100-80g",  "gpu", peak_tflops=312,  hbm_bw_tbs=2.00, mem_gb=80,  base_price_hr=1.40),
        SiliconProfile("l40s",      "gpu", peak_tflops=362,  hbm_bw_tbs=0.864, mem_gb=48, base_price_hr=0.95,
                       prefill_overhead_ms=0.0),    # uncalibrated; fit per run from batch=1 traces
        SiliconProfile("cpu-spr",   "cpu", peak_tflops=8,    hbm_bw_tbs=0.30, mem_gb=512, base_price_hr=0.60,
                       mfu=0.30, bw_eff=0.60),
        # TPU: systolic-array (non-SIMT) accelerator. The cross-architecture
        # generalization test -- if the roofline (peak compute + bandwidth)
        # predicts a TPU, the physics is not GPU-specific. Single-chip VMs
        # (v6e-1 / v5e-1) mirror the single-accelerator GPU runs. Specs: v6e
        # (Trillium) 918 BF16 TFLOPS / 32GB / 1.64 TB/s; v5e 197 / 16GB / 0.819.
        # prefill_overhead_ms left 0.0 -> fit from batch=1 traces post-run (XLA
        # first-compile is warm-up, excluded; steady-state floor is what we fit).
        SiliconProfile("tpu-v6e",   "tpu", peak_tflops=918,  hbm_bw_tbs=1.64, mem_gb=32,  base_price_hr=4.20),
        SiliconProfile("tpu-v5e",   "tpu", peak_tflops=197,  hbm_bw_tbs=0.819, mem_gb=16, base_price_hr=1.20),
    ]
}
