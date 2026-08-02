## v0.5.0

- feat(traces): `TraceRecord.source` ("measured" | "mock"), schema 2. A mock
  trace and a hardware trace were previously indistinguishable on disk while
  the contribution path was an open pull request, so the corpus could be
  corrupted by accident. Schema-1 files load as "measured" (this repo's own
  P0 traces are real); contributions must state it explicitly.
- feat(sounding): stamp source on every record; `provenance_of()` refuses a
  trace set mixing measured and mock.
- feat(validate): report provenance in the header, warn loudly on mock.
- feat(fit_overhead): `python -m bench.fit_overhead` fits the fixed prefill
  floor from batch-1 cells. The floor belongs to one (accelerator, driver,
  server, config) tuple and is not spec-predictable, so profiles ship 0.0 and
  each run fits its own. Refuses flat sweeps and flags negative fits.
- feat(check_contributed): CI gate rejecting mock or unlabelled contributions.
- test: 16 new tests covering provenance, back-compatibility and the fitter.

# Changelog

## 0.1.0 (berth)
- clean-room export as berth: package renamed, metric renamed to
  placement_premium, fresh history

### prior internal iterations (pre-export)

## 0.7.0
- feat: quadratic causal-attention prefill FLOPs + context-dependent decode
  attention FLOPs (fixes long-context TTFT underprediction; MI300X/70B
  implied-mfu bias 19% -> <2% in mock validation)
- feat: bootstrap 95% CIs on calibrated efficiency factors
- feat: harness records server-reported prompt tokens (usage), not the
  request-side heuristic
- feat: bench/microbench.py — GEMM + bandwidth ceiling probes for the
  fitted <= microbenched <= peak physics gate
- chore: ruff lint clean; ruff + pyright in CI

## 0.6.1
- fix: migrate held placements in policy-constraint violation inside the
  hysteresis band; fix: warm-up discard broken by randomized sweep order

## 0.6
- feat: physics validation report (term-by-term, identifiability gates);
  randomized sweep order (thermal-drift confound)

## 0.5
- feat: bench harness (mock + real vLLM modes); production plan

## 0.2 - 0.4
- calibration (inversion + blind recovery), M/M/c tail-aware sizing,
  per-class fits, drift detection

## 0.1
- placement primitives over roofline model, simulated market backend
