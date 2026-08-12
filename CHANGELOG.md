## v0.6.0

The control plane. berth predicted; it now decides, watches, and proves.

**Added**

- `berth.place`, the decision record: what is recommended, what it beat, by
  how much, and whether that margin clears the uncertainty plus the cost of
  moving. A placement missing the bound is excluded rather than ranked
  cheaply, because its cost per compliant token is undefined rather than high.
- `berth.holdout`, the reference implementation of the Placement Holdout
  Protocol. Assignment, declaration, period bookkeeping, four stationarity
  checks, warm-up detection, circuit breaker.
- `berth.receipt`, two-leg settlement into a conforming record. Holdout cost
  is its own line, a negative period carries forward, and a tripped breaker
  voids rather than counting as a loss.
- `berth.agent`, the loop: watch, detect, re-estimate, propose. State stops it
  repeating a proposal or re-asking after a rejection.
- `berth.watch`, four sources: model registries by commit, serving-stack
  releases, provider prices with an epsilon, corpus additions. An unreachable
  source is recorded rather than swallowed.
- `berth.github`, opens the pull request and refuses everything else. Cannot
  merge, cannot write to a default branch, cannot touch an undeclared path.
- `berth.declaration`, `.berth/classes.yaml` in the customer's own repository.
  Declared axes and chosen axes are separated, and narrowing the search
  requires a reason.
- `berth.status`, `.berth/STATUS.md`, every class in one file.
- `berth.ledger`, realized, available and foregone savings, never summed.
- `berth.versus`, self-host or API on one axis.
- `berth.quantities`, typed ceilings that make two unit defects impossible at
  the call site.
- Price basis as a search dimension: spot, on-demand and reserved as distinct
  candidates with distinct admissibility.
- Device power in the trace schema, at the device boundary only.
- CLI: `place`, `pilot`, `versus`, `holdout`.

**Changed**

- Key-value bandwidth varies with concurrency where a device has been measured
  to have two access patterns. One card holds 3,854 GB/s to batch 4 and falls
  to 633 by batch 16 while both NVIDIA cards measured stay flat. The estimator
  previously used one constant and was five times optimistic at batch 16.
- `MEASURED` derives from the corpus rather than being frozen at the two P0
  cards.
- The audit refuses a file only on physical impossibility. Everything else
  prints and passes.
- `bench/holdout.py` is now `bench/crossval.py`. It cross-validates the
  estimator; `berth.holdout` is the commercial protocol, and one name for two
  unrelated things was a collision waiting to happen.

**Fixed**

- Eleven instrument defects, registered in `DEFECTS.md` with the mechanism,
  how each was caught, and the test that fails if it returns. Four let bad
  data through. Seven rejected good data, which is the more dangerous
  direction because a false positive is silent.

## v0.5.0

**Trace provenance.** `TraceRecord.source` is "measured" or "mock", and
`bench.sounding` stamps it on every record. A mock trace and a hardware trace
were previously indistinguishable on disk while the contribution path was an
open pull request, so the corpus could be corrupted by accident.
`provenance_of()` refuses a trace set mixing the two, `bench.validate` reports
provenance in its header and warns loudly on mock, and
`bench.check_contributed` gates contributions in CI.

**Schema 3.** Two branches independently minted a schema 2, one adding
`w_bytes`/`kv_bytes` for quantisation and one adding `source`. Neither is a
superset, so v2 on disk is ambiguous. v3 carries both, and the loader
back-fills whichever half is absent for anything below v3. The 60 P0 traces
load unchanged.

**Quantisation in traces.** `w_bytes` and `kv_bytes` are recorded per cell and
`TraceRecord.signature()` resolves a quantised model spec, so an fp8 cell is
inverted against an fp8 signature rather than silently absorbing the dtype
delta into the fitted mfu/bw_eff.

**`bench.fit_overhead`.** Fits the fixed prefill floor from batch-1 cells by
Theil-Sen regression. The floor belongs to one (accelerator, driver, server,
config) tuple and is not spec-predictable: 74.6 ms on an L40S, 54.6 ms on an
H100 PCIe. Profiles ship 0.0 and each run fits its own. Refuses flat sweeps and
flags negative fits.

**CLI.** `berth estimate`, `berth premium`, `berth list`, installed as the
`berth` console script. Every line tags its silicon MEASURED or prior.

**sounding.** The measurement harness is renamed from `bench.run_sweep`, which
is retained as a deprecation shim.

**Fleet.** Adds H100 PCIe (measured), B200 with a native fp4 path, and TPU
v6e/v5e profiles. Adds Mixtral-8x7B and Qwen3-30B-A3B model specs.

**Packaging.** Published to PyPI as `berth-placement`; the console script is
still `berth`. License declared as an SPDX string, `project.urls` added.

73 tests passing.

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
- feat: bench/microbench.py , GEMM + bandwidth ceiling probes for the
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
