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
