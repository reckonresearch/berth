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
    bytes_per_param: float = 2.0    # fp16 weights; 1.0 for fp8/int8, 0.5 for fp4/int4
    bytes_per_kv: float = 2.0       # fp16 KV cache
    # Attention family sets the KV-cache STRUCTURE, not just its size. "gqa"
    # covers MHA/GQA (MHA == GQA with n_kv_heads == n_heads). "mla" (DeepSeek,
    # Kimi, Mistral-Large-3) caches a compressed latent, a fundamentally
    # different footprint — see kv_bytes_per_token. Baking GQA in as the only
    # form was a convention, not a law; MLA is half the open frontier.
    attn: str = "gqa"
    kv_lora_rank: int = 0           # MLA: KV compression dim (0 for GQA)
    qk_rope_head_dim: int = 0       # MLA: decoupled RoPE key dim (0 for GQA)
    # Explicit quant format labels. bytes_per_* set the physics (footprint,
    # bandwidth); these name the FORMAT for provenance, since byte-count alone
    # can't tell fp8 from int8 (both 1 byte) or fp4 from int4. Empty -> derive
    # a coarse label from the byte count.
    weight_fmt: str = ""
    kv_fmt: str = ""

    ATTN_FAMILIES = ("gqa", "mla", "sparse")

    def __post_init__(self):
        # Fail fast and loud: an MLA spec with no latent dim would silently fall
        # back to a wrong number. Validate invariants at construction.
        if self.attn not in self.ATTN_FAMILIES:
            raise ValueError(f"{self.name}: unknown attn family {self.attn!r} "
                             f"(known: {self.ATTN_FAMILIES})")
        if self.attn == "mla" and self.kv_lora_rank <= 0:
            raise ValueError(f"{self.name}: attn='mla' requires kv_lora_rank > 0")

    @property
    def weight_bytes(self) -> float:
        return self.total_params_b * 1e9 * self.bytes_per_param

    @property
    def kv_bytes_per_token(self) -> float:
        if self.attn == "sparse":
            # Sparse attention (DeepSeek-V4 CSA/HCA, etc.) selects a runtime
            # top-k of KV entries; the resident footprint is NOT a closed-form
            # per-token constant and depends on the selection budget. Fail loud
            # rather than silently reuse a wrong (MLA or GQA) formula — this is a
            # THIRD KV regime, not covered yet. See FLAG: sparse-KV term.
            raise NotImplementedError(
                f"{self.name}: sparse-attention KV footprint not modeled "
                "(runtime top-k selection); do not estimate until a sparse term lands")
        if self.attn == "mla":
            # MLA caches ONE low-rank latent (kv_lora_rank) plus the decoupled
            # RoPE key (qk_rope_head_dim) per layer — reconstructing per-head K/V
            # on the fly via absorbed up-projections. No factor of 2 (a single
            # shared latent, not separate K and V) and no n_kv_heads (the latent
            # is head-agnostic). ~50x smaller than the GQA form for DeepSeek's
            # shape; the GQA formula overpredicts KV by that factor. Oracle: the
            # DeepSeek reference impl caches kv_cache[kv_lora_rank] + pe_cache
            # [qk_rope_head_dim] — this formula mirrors those two buffers.
            return self.n_layers * (self.kv_lora_rank + self.qk_rope_head_dim) * self.bytes_per_kv
        # GQA/MHA: separate K and V, per KV head, per layer.
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.bytes_per_kv

    @staticmethod
    def _fmt(bytes_per: float, explicit: str) -> str:
        if explicit:
            return explicit
        return {2.0: "bf16", 1.0: "8bit", 0.5: "4bit"}.get(bytes_per, f"{bytes_per * 8:g}bit")

    @property
    def quant_label(self) -> str:
        # Provenance: a fp8 measurement is not comparable to a bf16 one, and
        # fp8 != int8 even at equal byte-count. Prefer the explicit format.
        return f"w:{self._fmt(self.bytes_per_param, self.weight_fmt)}/kv:{self._fmt(self.bytes_per_kv, self.kv_fmt)}"

    def quantized(self, w_bytes: float, kv_bytes: float | None = None,
                  w_fmt: str = "", kv_fmt: str = "") -> "ModelSpec":
        """Return a quantized variant. KV is often kept higher-precision than
        weights (e.g. fp8 weights, bf16 KV), so kv_bytes is set independently.
        w_fmt/kv_fmt name the format explicitly (fp8 vs int8) for provenance."""
        from dataclasses import replace
        return replace(self, bytes_per_param=w_bytes,
                       bytes_per_kv=self.bytes_per_kv if kv_bytes is None else kv_bytes,
                       weight_fmt=w_fmt, kv_fmt=kv_fmt or self.kv_fmt)


# Presets with real architecture numbers (GQA models — KV heads << attn heads).
MODELS: dict[str, ModelSpec] = {
    m.name: m
    for m in [
        ModelSpec("llama3-8b",   total_params_b=8,   active_params_b=8,   n_layers=32, n_kv_heads=8, head_dim=128, n_heads=32),
        ModelSpec("llama3-70b",  total_params_b=70,  active_params_b=70,  n_layers=80, n_kv_heads=8, head_dim=128, n_heads=64),
        ModelSpec("qwen3-235b-moe", total_params_b=235, active_params_b=22, n_layers=94, n_kv_heads=4, head_dim=128, n_heads=64),
        # Single-card MoE cell: 235B needs 470GB bf16 (>1 MI300X); Mixtral fits (93GB) and
        # still breaks the dense-matmul assumption (active 12.9B != total 46.7B). 8 experts, top-2.
        ModelSpec("mixtral-8x7b", total_params_b=46.7, active_params_b=12.9, n_layers=32, n_kv_heads=8, head_dim=128, n_heads=32),
        # MLA MoE (DeepSeek-lineage). KV is a compressed latent, NOT per-head K/V:
        # kv_lora_rank=512 + qk_rope_head_dim=64 per layer. V3 and R1 share the
        # arch; head_dim/n_heads feed only the attention-FLOPs proxy (KV ignores
        # them for attn='mla'). Kimi K2.6 and Mistral-Large-3 are also MLA — add
        # them once their published kv_lora_rank is confirmed (don't fabricate).
        ModelSpec("deepseek-v3", total_params_b=671, active_params_b=37, n_layers=61, n_kv_heads=128, head_dim=128, n_heads=128, attn="mla", kv_lora_rank=512, qk_rope_head_dim=64),
        ModelSpec("deepseek-r1", total_params_b=671, active_params_b=37, n_layers=61, n_kv_heads=128, head_dim=128, n_heads=128, attn="mla", kv_lora_rank=512, qk_rope_head_dim=64),
        # DeepSeek-V2-Lite: independent MLA config (27 layers, same rank 512 / rope 64,
        # 15.7B/2.4B). Two purposes: (1) de-circularizes the MLA KV test — a different
        # layer count proves the term scales with L, not a hardcoded constant; (2) fits
        # ONE MI300X (~31GB), so measured-MLA validation costs $2.80/hr, not an 8x node.
        ModelSpec("deepseek-v2-lite", total_params_b=15.7, active_params_b=2.4, n_layers=27, n_kv_heads=16, head_dim=128, n_heads=16, attn="mla", kv_lora_rank=512, qk_rope_head_dim=64),
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
