"""Workload specification and the `profile` primitive.

`profile()` reduces a serving workload to a compute signature: the small set
of numbers the estimator needs. This is the analogue of Tinker letting you
swap models by changing one string — a WorkloadSpec is that string, expanded.

MoE handling: compute and bandwidth per token scale with ACTIVE params;
memory footprint scales with TOTAL params. Conflating these is the single
most common error in heterogeneous cost models.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    total_params_b: float       # billions — sets memory footprint
    active_params_b: float      # billions — sets per-token compute/bandwidth
    n_layers: int
    n_kv_heads: int
    head_dim: int
    n_heads: int                    # query heads; d_attn = n_heads * head_dim
    bytes_per_param: float = 2.0    # fp16 weights; 1.0 for int8, 0.5 for int4
    bytes_per_kv: float = 2.0       # fp16 KV cache

    @property
    def weight_bytes(self) -> float:
        return self.total_params_b * 1e9 * self.bytes_per_param

    @property
    def kv_bytes_per_token(self) -> float:
        # K and V, per layer: n_kv_heads * head_dim elements each.
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.bytes_per_kv


# Presets with real architecture numbers (GQA models — KV heads << attn heads).
MODELS: dict[str, ModelSpec] = {
    m.name: m
    for m in [
        ModelSpec("llama3-8b",   total_params_b=8,   active_params_b=8,   n_layers=32, n_kv_heads=8, head_dim=128, n_heads=32),
        ModelSpec("llama3-70b",  total_params_b=70,  active_params_b=70,  n_layers=80, n_kv_heads=8, head_dim=128, n_heads=64),
        ModelSpec("qwen3-235b-moe", total_params_b=235, active_params_b=22, n_layers=94, n_kv_heads=4, head_dim=128, n_heads=64),
    ]
}


@dataclass(frozen=True)
class WorkloadSpec:
    """What the user actually knows about their traffic."""
    model: ModelSpec
    avg_prompt_tokens: int = 1024
    avg_output_tokens: int = 256
    target_batch: int = 16          # concurrent sequences per replica
    p99_ttft_ms: float | None = None    # SLO: time-to-first-token ceiling
    p99_tpot_ms: float | None = None    # SLO: time-per-output-token ceiling
    arrival_rps: float | None = None    # traffic; None -> single-replica mode


@dataclass(frozen=True)
class ComputeSignature:
    """Output of profile(): everything the estimator needs, nothing else."""
    model: ModelSpec
    batch: int
    avg_context: int                # avg tokens resident in KV per sequence
    prefill_flops_per_req: float    # compute cost of one prompt
    decode_flops_per_token: float   # compute cost of one output token (per seq)
    decode_ai: float                # arithmetic intensity of decode (FLOPs/byte)
    p99_ttft_ms: float | None
    p99_tpot_ms: float | None
    arrival_rps: float | None
    avg_output_tokens: int


def profile(w: WorkloadSpec) -> ComputeSignature:
    m = w.model
    avg_context = w.avg_prompt_tokens + w.avg_output_tokens // 2
    d_attn = m.n_heads * m.head_dim
    # Linear (weights) term + quadratic causal-attention term. The L^2 term
    # was the audit's top physics gap: linear-only TTFT underpredicts at long
    # context, exactly where the RAG workload class lives.
    prefill_flops = (2.0 * m.active_params_b * 1e9 * w.avg_prompt_tokens
                     + 2.0 * m.n_layers * (w.avg_prompt_tokens ** 2) * d_attn)
    decode_flops = (2.0 * m.active_params_b * 1e9
                    + 4.0 * m.n_layers * avg_context * d_attn)

    # Decode arithmetic intensity at this batch size: weights are read once
    # per step and amortized across the batch; KV reads are per-sequence.
    active_weight_bytes = m.active_params_b * 1e9 * m.bytes_per_param
    bytes_per_step = active_weight_bytes + w.target_batch * avg_context * m.kv_bytes_per_token
    flops_per_step = decode_flops * w.target_batch
    decode_ai = flops_per_step / bytes_per_step

    return ComputeSignature(
        model=m,
        batch=w.target_batch,
        avg_context=avg_context,
        prefill_flops_per_req=prefill_flops,
        decode_flops_per_token=decode_flops,
        decode_ai=decode_ai,
        p99_ttft_ms=w.p99_ttft_ms,
        p99_tpot_ms=w.p99_tpot_ms,
        arrival_rps=w.arrival_rps,
        avg_output_tokens=w.avg_output_tokens,
    )
