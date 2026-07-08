"""Tests: estimator physics sanity, SLO filtering, placement, migration hysteresis."""


import pytest

from berth import (
    FLEET,
    MODELS,
    PlacementClient,
    PlacementPolicy,
    SimBackend,
    WorkloadSpec,
    min_cost,
)
from berth.estimate import estimate
from berth.workload import ModelSpec, profile


def sig_for(model="llama3-70b", **kw):
    return profile(WorkloadSpec(model=MODELS[model], **kw))


# ---------- estimator physics ----------

def test_kv_bytes_per_token_llama70b():
    # Known figure: Llama-3-70B GQA KV cache ~0.32 MB/token at fp16.
    assert MODELS["llama3-70b"].kv_bytes_per_token == pytest.approx(327_680)


def test_70b_needs_multiple_h100s():
    e = estimate(sig_for(), FLEET["h100-sxm"], FLEET["h100-sxm"].base_price_hr)
    assert e.feasible and e.n_devices >= 2  # 140GB weights alone > 80GB


def test_70b_fits_single_mi300x_or_two():
    e = estimate(sig_for(target_batch=4, avg_prompt_tokens=512),
                 FLEET["mi300x"], FLEET["mi300x"].base_price_hr)
    assert e.feasible and e.n_devices <= 2


def test_small_batch_decode_is_memory_bound():
    e = estimate(sig_for(target_batch=1), FLEET["h100-sxm"], 3.20)
    assert e.bound == "memory"  # batch-1 decode AI << H100 ridge point


def test_decode_bandwidth_math_batch1():
    # Batch-1 TPOT should ≈ active weight bytes / effective bandwidth.
    s = sig_for(model="llama3-8b", target_batch=1, avg_prompt_tokens=128, avg_output_tokens=64)
    hw = FLEET["h100-sxm"]
    e = estimate(s, hw, hw.base_price_hr)
    weight_t = (8e9 * 2) / (hw.hbm_bw_tbs * 1e12 * hw.bw_eff)
    assert e.tpot_ms == pytest.approx(weight_t * 1e3, rel=0.10)  # KV adds a little


def test_moe_memory_vs_compute_split():
    # Qwen MoE: big memory footprint, small active compute -> cheap per token
    # relative to a dense model of equal total size would be.
    moe = estimate(sig_for(model="qwen3-235b-moe", target_batch=8),
                   FLEET["mi300x"], FLEET["mi300x"].base_price_hr)
    assert moe.feasible
    assert moe.n_devices >= 3          # 470GB of weights
    assert moe.tpot_ms < 25            # but only 22B active params per token


def test_mixtral_single_card_moe_cell():
    # Single-card MoE fallback: Mixtral-8x7B (46.7B total / 12.9B active) must fit
    # ONE MI300X (93GB bf16 << 192GB) so the model-shape cell is runnable without TP.
    e = estimate(sig_for(model="mixtral-8x7b", target_batch=8),
                 FLEET["mi300x"], FLEET["mi300x"].base_price_hr)
    assert e.feasible and e.n_devices == 1     # 93GB weights fit one 192GB card
    # Decode keyed on ACTIVE (12.9B), not total: TPOT well under a dense-46.7B model.
    dense_46b = MODELS["mixtral-8x7b"].active_params_b == 12.9
    assert dense_46b and e.tpot_ms < 20        # active-param bandwidth, not total


def test_mla_kv_is_compressed_latent_not_per_head():
    # FLAG 1: MLA caches one latent (kv_lora_rank) + rope key per layer, not
    # per-head K/V. DeepSeek-V3 published config: 61 layers, rank 512, rope 64.
    ds = MODELS["deepseek-v3"]
    assert ds.kv_bytes_per_token == 61 * (512 + 64) * 2      # 70,272 bytes/token @ bf16
    # The bug we're fixing: the GQA/MHA formula would overpredict by ~57x.
    mha_naive = 2 * ds.n_layers * ds.n_kv_heads * ds.head_dim * ds.bytes_per_kv
    assert ds.kv_bytes_per_token < mha_naive / 50


def test_mla_spec_fails_loud_without_latent():
    # An MLA spec with no latent dim would silently emit a wrong number.
    with pytest.raises(ValueError, match="kv_lora_rank"):
        ModelSpec("bad-mla", total_params_b=1, active_params_b=1, n_layers=1,
                  n_kv_heads=1, head_dim=1, n_heads=1, attn="mla")


def test_quant_provenance_labels_and_scales():
    # FLAG 2: quant is a comparability axis; weights and KV quantize independently.
    ds = MODELS["deepseek-v3"]
    assert ds.quant_label == "w:bf16/kv:bf16"
    fp8 = ds.quantized(1.0)                       # fp8 weights, KV left at bf16
    assert fp8.weight_bytes == ds.weight_bytes / 2
    assert fp8.quant_label == "w:fp8/kv:bf16"
    assert ds.quantized(1.0, 1.0).quant_label == "w:fp8/kv:fp8"


def test_deepseek_mla_fits_one_mi300x_node():
    ds = MODELS["deepseek-v3"]
    e_bf16 = estimate(sig_for(model="deepseek-v3", target_batch=8),
                      FLEET["mi300x"], FLEET["mi300x"].base_price_hr)
    assert e_bf16.feasible and e_bf16.n_devices <= 8   # 1.34TB bf16 on an 8x192GB node
    e_fp8 = estimate(profile(WorkloadSpec(model=ds.quantized(1.0), target_batch=8)),
                     FLEET["mi300x"], FLEET["mi300x"].base_price_hr)
    assert e_fp8.n_devices < e_bf16.n_devices          # fp8 halves the weight footprint


def test_infeasible_when_too_large_for_node():
    # int4 would fit; fp16 235B + heavy KV on 48GB cards blows the 8-device cap.
    e = estimate(sig_for(model="qwen3-235b-moe", target_batch=64, avg_prompt_tokens=8192),
                 FLEET["l40s"], 0.95)
    assert not e.feasible and "devices" in e.reason


# ---------- SLO constraints ----------

def test_tpot_slo_filters_cpu():
    s = profile(WorkloadSpec(model=MODELS["llama3-8b"], p99_tpot_ms=50, target_batch=4))
    e = estimate(s, FLEET["cpu-spr"], 0.60)
    assert not e.feasible and "TPOT" in e.reason


# ---------- client: estimate / place / migrate ----------

def test_estimate_annotates_placement_premium():
    client = PlacementClient(SimBackend(seed=1))
    ests = client.estimate(sig_for())
    feas = [e for e in ests if e.feasible]
    assert min(e.placement_premium for e in feas) == pytest.approx(0.0)
    assert max(e.placement_premium for e in feas) > 0.0


def test_place_selects_min_cost_feasible():
    client = PlacementClient(SimBackend(seed=1))
    sig = sig_for()
    h = client.place(sig, PlacementPolicy(objective=min_cost))
    best = min((e for e in client.estimate(sig) if e.feasible), key=min_cost)
    assert h.silicon == best.silicon


def test_place_respects_constraints():
    client = PlacementClient(SimBackend(seed=1))
    sig = sig_for()
    h = client.place(sig, PlacementPolicy(
        objective=min_cost,
        constraints=(lambda e: e.ttft_ms < 400,),
    ))
    assert h.estimate.ttft_ms < 400


def test_migrate_holds_within_hysteresis():
    backend = SimBackend(seed=1)
    client = PlacementClient(backend)
    h = client.place(sig_for(), PlacementPolicy(objective=min_cost, min_improvement=0.15))
    home = h.silicon
    backend.shock(home, 1.05)  # 5% bump: inside hysteresis band
    h2 = client.migrate(h)
    assert h2.silicon == home


def test_migrate_moves_on_price_shock():
    backend = SimBackend(seed=1)
    client = PlacementClient(backend)
    h = client.place(sig_for(), PlacementPolicy(objective=min_cost, min_improvement=0.15))
    home = h.silicon
    backend.shock(home, 2.5)  # spot spike on current home
    h2 = client.migrate(h)
    assert h2.silicon != home
    assert h2.estimate.cost_per_mtok < h.estimate.cost_per_mtok * 2.5


def test_migrate_releases_old_allocation():
    backend = SimBackend(seed=1, capacity_per_class=8)
    client = PlacementClient(backend)
    h = client.place(sig_for(), PlacementPolicy(objective=min_cost))
    home, n = h.silicon, h.estimate.n_devices
    cap_before = backend._capacity[home]
    backend.shock(home, 2.5)
    client.migrate(h)
    assert backend._capacity[home] == cap_before + n  # devices returned to pool


def test_migrate_forces_move_on_constraint_violation():
    # Price shock pushes current cost over the user's ceiling, but the
    # improvement to the next-best is inside hysteresis. Constraint violation
    # must override hysteresis.
    backend = SimBackend(seed=1)
    client = PlacementClient(backend)
    policy = PlacementPolicy(objective=min_cost,
                             constraints=(lambda e: e.cost_per_mtok < 3.2,),
                             min_improvement=0.15)
    h = client.place(sig_for(), policy)
    assert h.silicon == "mi300x"
    backend.shock("mi300x", 1.6)          # cost ~3.30 > 3.2; a100 ~3.06 < 3.2
    h2 = client.migrate(h)
    assert h2.silicon != "mi300x"
    assert h2.estimate.cost_per_mtok < 3.2
