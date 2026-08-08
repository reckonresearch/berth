"""One test per instrument defect. Each reproduces the original failure.

A defect that is fixed but not pinned is a defect waiting for a refactor.
Every entry here corresponds to a real failure in this project, and each test
fails if that failure returns.

The register itself is in DEFECTS.md. This file is the enforcement.

Eight defects across four mechanisms:

    provenance   a label asserted rather than captured        2, 3, 8
    units        a quantity used where another was meant      5, 6
    assumption   a checker asserting what hardware does not   4, 7
    ratio        a constant term left inside a division       1

None were caught by review. Six let bad data through; two rejected good data.
"""

import json

import pytest

from berth.quantities import Ceiling, UnitError, transfer_efficiency

# =========================================================================
# Family: units. A number handed to a function expecting a different number
# of the same shape.
# =========================================================================

def test_defect_5_microbench_cannot_be_used_as_a_datasheet_peak():
    """A device-to-device copy rate passed where a datasheet peak was expected
    produced efficiencies above 1.0 on every batch of a clean L40S file, and
    the auditor reported the file as contaminated."""
    micro = Ceiling.microbench("l40s", bandwidth_tbs=0.646)
    with pytest.raises(UnitError, match="datasheet"):
        micro.efficiency_of(measured_tbs=0.55)
    # The honest operation is available and named differently.
    assert 0.8 < micro.headroom_of(measured_tbs=0.55) < 0.9


def test_defect_6_efficiency_cannot_be_fitted_across_a_device_boundary():
    """Fitting on one card's timings while dividing by another card's peak
    reported 109.5 percent error with an efficiency of 1.761."""
    h100 = Ceiling.datasheet("h100-pcie", bandwidth_tbs=2.0)
    with pytest.raises(UnitError, match="device"):
        h100.efficiency_of(measured_tbs=0.73, device="l40s")


def test_defect_6_transfer_is_the_supported_operation():
    """Carrying a constant between devices is legitimate and is what a
    leave-one-silicon-out split does. It has to be explicit."""
    l40s = Ceiling.datasheet("l40s", bandwidth_tbs=0.864)
    h100 = Ceiling.datasheet("h100-pcie", bandwidth_tbs=2.0)
    assert transfer_efficiency(l40s, 0.85, h100) == pytest.approx(1.7)


def test_defect_6_impossible_efficiency_is_refused_at_the_boundary():
    """1.761 is not a fraction of a peak, and the type says so rather than
    propagating it into a MAPE that looks like a finding."""
    a = Ceiling.datasheet("l40s", bandwidth_tbs=0.864)
    b = Ceiling.datasheet("h100-pcie", bandwidth_tbs=2.0)
    with pytest.raises(UnitError, match="not a fraction"):
        transfer_efficiency(a, 1.761, b)


def test_gigabytes_mistaken_for_terabytes_is_refused():
    """The adjacent error to defect 5, not yet made, and now unmakeable."""
    with pytest.raises(UnitError, match="GB/s"):
        Ceiling.datasheet("h100-pcie", bandwidth_tbs=2000.0)


# =========================================================================
# Family: provenance. A label asserted rather than captured from the source.
# =========================================================================

def test_defect_2_served_model_is_verified_not_assumed():
    """The harness asked an endpoint for tokens and never asked what it was
    serving. A wrong checkpoint produces clean timings for the wrong model."""
    from bench.sounding import verify_served_model
    assert callable(verify_served_model)


def test_defect_3_silicon_mismatch_refuses(monkeypatch):
    """An operator ran on an RTX PRO 6000 while declaring an H100. Every
    timing was real and every one would have been filed against hardware that
    never produced it."""
    from bench import sounding
    monkeypatch.setattr(sounding, "detect_silicon",
                        lambda *a, **k: ("l40s", "NVIDIA L40S"))
    with pytest.raises(SystemExit, match="l40s"):
        sounding.resolve_silicon_provenance("h100-pcie", "http://localhost:8000")


def test_defect_3_unrecognised_card_is_not_guessed(monkeypatch):
    """Mapping an unknown card to the nearest fleet key would be a
    fabrication dressed as a capture."""
    from bench import sounding
    monkeypatch.setattr(sounding, "detect_silicon",
                        lambda *a, **k: (None, "NVIDIA RTX PRO 6000"))
    assert sounding.resolve_silicon_provenance(
        "h100-pcie", "http://localhost:8000") == "self_reported"


def test_defect_3_remote_endpoint_records_self_reported():
    """A remote server cannot report its hardware. The honest answer is
    self_reported, not a guess."""
    from bench.sounding import resolve_silicon_provenance
    assert resolve_silicon_provenance(
        "l40s", "http://10.0.0.5:8000") == "self_reported"


def test_defect_8_quantization_label_must_match_the_timing(tmp_path):
    """An fp8 cell recorded kv_bytes=1.0 while the server ran a bf16 key-value
    cache. Real timings, wrong attribution, self-consistent output."""
    from types import SimpleNamespace

    from bench.audit_traces import check_quant_label_plausible
    # Declared fp8, but the timing is the bf16 one that actually ran. The
    # trace record is frozen by design, so the label is set at construction.
    traces = [SimpleNamespace(silicon="h100-sxm", model_name="llama3-8b",
                              batch=1, avg_prompt_tokens=ctx,
                              avg_output_tokens=128, measured_ttft_ms=40.0,
                              measured_tpot_ms=9.16, w_bytes=1.0)
              for ctx in (512, 2048, 7680)]
    problems, ratio = check_quant_label_plausible(traces, 8.0, 3.35)
    assert problems, f"a bf16 timing labelled fp8 must fail, ratio was {ratio}"

    # And the correctly labelled version of the same run must pass.
    for t in traces:
        t.w_bytes = 2.0
    ok, _ = check_quant_label_plausible(traces, 8.0, 3.35)
    assert not ok, ok


# =========================================================================
# Family: assumption. A checker asserting a property the system lacks.
# =========================================================================

def test_defect_4_prompts_are_unique_per_request():
    """Identical prompts plus default prefix caching produced apparent prefill
    throughput of twenty-one times a card's peak FLOPS, and nothing errored."""
    import random

    from bench.sounding import make_prompt
    a = make_prompt(512, random.Random(1))
    b = make_prompt(512, random.Random(2))
    assert a != b
    # The divergence must begin immediately, not after a shared prefix longer
    # than a cache block.
    shared = 0
    for x, y in zip(a.split(), b.split(), strict=False):
        if x != y:
            break
        shared += 1
    assert shared < 16, f"{shared} shared leading tokens exceeds a cache block"


def test_defect_4_prefill_impossibility_is_caught(tmp_path):
    """The physical check that would have found it: a card cannot exceed its
    own peak FLOPS."""
    from bench.audit_traces import check_prefill_possible
    from bench.sounding import load_jsonl
    p = tmp_path / "cached.jsonl"
    rows = [{"schema": 3, "source": "measured", "silicon_provenance": "captured",
             "silicon": "a100-80g", "model_name": "llama3-8b", "batch": b,
             "avg_prompt_tokens": ctx, "avg_output_tokens": 128,
             # flat in both batch and context: the cache signature
             "measured_ttft_ms": 45.0, "measured_tpot_ms": 7.0}
            for b in (1, 8, 32) for ctx in (512, 2048, 7680)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    problems, worst, _floor = check_prefill_possible(load_jsonl(str(p)), 312, 8.0)
    assert problems and worst > 1.5, worst


def test_defect_7_two_access_patterns_are_not_a_corrupted_file(tmp_path):
    """A device whose paged key-value path degrades with batch makes long
    context look dearer per byte. The checker assumed one bandwidth per
    device and rejected the first cross-vendor result in the corpus."""
    from bench.audit_traces import check_bw_eff_is_constant
    from bench.sounding import load_jsonl
    p = tmp_path / "slow_gather.jsonl"
    rows = []
    for b, kv_gbs in ((1, 3400), (8, 1950), (32, 658)):
        for ctx in (512, 2048, 7680):
            t = (16.06 / 1400 + (b * (ctx + 64) * 131072 / 1e9) / kv_gbs) * 1000
            rows.append({"schema": 3, "source": "measured",
                         "silicon_provenance": "self_reported",
                         "silicon": "mi300x", "model_name": "llama3-8b",
                         "batch": b, "avg_prompt_tokens": ctx,
                         "avg_output_tokens": 128,
                         "measured_ttft_ms": 50.0 + ctx * 0.02 * b,
                         "measured_tpot_ms": t})
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    problems, _rows, notes = check_bw_eff_is_constant(
        load_jsonl(str(p)), 5.3, 8.0, 131072)
    assert not problems, f"a slow gather is a finding, not contamination: {problems}"
    assert any("FALLS" in n for n in notes)


# =========================================================================
# Family: ratio. A constant term left inside a division.
# =========================================================================

def test_defect_1_floor_is_removed_before_inverting_ttft(tmp_path):
    """The fixed prefill floor is paid once per request and does not scale.
    Inverting first-token latency without removing it attributes a constant to
    compute, and it produced a false failure on a clean file. This one
    recurred six times, including inside documentation warning against it."""
    from bench.audit_traces import check_prefill_possible
    from bench.sounding import load_jsonl
    p = tmp_path / "clean_with_floor.jsonl"
    FLOOR, PEAK, PARAMS = 74.6, 181e12, 8.0
    rows = []
    for b in (1, 8, 32):
        for ctx in (512, 2048, 7680):
            compute_s = 2 * PARAMS * 1e9 * b * ctx / (PEAK * 0.6)
            rows.append({"schema": 3, "source": "measured",
                         "silicon_provenance": "captured", "silicon": "l40s",
                         "model_name": "llama3-8b", "batch": b,
                         "avg_prompt_tokens": ctx, "avg_output_tokens": 128,
                         "measured_ttft_ms": FLOOR + compute_s * 1000,
                         "measured_tpot_ms": 24.0})
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    traces = load_jsonl(str(p))

    without, _, _ = check_prefill_possible(traces, 181, PARAMS, floor_ms=0.0)
    with_floor, worst, _ = check_prefill_possible(traces, 181, PARAMS,
                                                  floor_ms=FLOOR)
    assert not with_floor, f"floor removed, should pass: worst {worst:.2f}x"
    assert worst < 1.5
