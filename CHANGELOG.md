## v0.8.0

The placement API, and a delivery path that was written and unreachable.

**Added**

- `berth.api`, placement decisions over HTTP. pilot delivers through a
  repository, which reaches every team that deploys from git and no team that
  does not. An orchestrator, a CI pipeline or a scheduler cannot read a pull
  request; this is the same decision as JSON.
- `POST /v1/place` returns the decision plus `act`, because a caller acting on
  this automatically has no chance to ask a follow-up question. Every figure
  carries whether it rests on a measurement or a specification sheet.
- `POST /v1/versus`, self-host or API, taking the rate cards from the caller.
  A price list we maintained would be stale the week after it shipped, and a
  wrong price is worse than none because it looks authoritative.
- `GET /v1/silicon`, the fleet and which cells are measured.
- A stated refusal on `/v1/completions`, `/v1/chat/completions` and
  `/v1/generate`. The refusal is a route rather than an absence, so a caller
  expecting a proxy gets the reason instead of a 404 they might read as a
  missing feature. berth returns placement decisions and never sits in the
  request path: a party that carries traffic cannot credibly rank the
  placements it carries traffic for.
- `berth-api` console script. Binds to loopback by default, because a
  placement API is an internal service and binding to all interfaces by
  default is how something ends up on the public internet because nobody
  passed a flag.

**Fixed**

- The execution layer was written, tested and reachable from nothing. Neither
  the agent loop nor the CLI imported it, so pilot proposed and never executed
  in any path a customer would run. Delivery is now injected into `run()` the
  same way the decision resolver and the trigger detector are, which also
  means a second delivery target is a new adapter rather than a fork.
- `MEASURED` in this build was still the frozen P0 pair.

**Notes**

- No web framework. The standard library only, because a placement decision
  should not oblige anyone to adopt a web stack, and because this has to run
  in a customer's environment as easily as in ours. Roughly two hundred lines
  against a dependency that would be forty thousand.

## v0.7.0

pilot executes. A placement system that only proposes is a recommendation
engine with extra steps: if every change waits on someone noticing a pull
request, the loop is as slow as the human in it, and the argument for adaptive
placement is that the answer changes faster than anyone checks.

**Added**

- `berth.execute`, the execution layer. Authority is declared once, in advance,
  in the customer's own repository, and that commit is the authorization. The
  human decision moves from approving a change to approving a class of change
  under stated conditions, which is the decision a human is good at.
- Four autonomy levels per class: `frozen`, `propose` (the default),
  `window`, `execute`.
- Six guardrails, every one of which can only refuse. Blast radius so one bad
  estimate cannot move a fleet. A rate limit so a class cannot oscillate.
  Blackout dates, because a freeze is real and a system that ignores one gets
  uninstalled. A minimum margin to act unattended, higher than the bar for
  proposing. A measurement requirement, so an unmeasured placement is never
  executed by default regardless of how good it looks. And automatic rollback.
- `GitHubClient.merge_proposal`, which merges a pull request pilot opened under
  a stated policy and writes the reason into the merge commit. A change that
  landed unattended says on its face which rule allowed it, and reverting it is
  reverting a commit.
- `sounding` and `pilot` as their own console commands. Reaching the agent
  through `berth pilot` read as the estimator's subcommand, which is the wrong
  shape for a separate product. The subcommands still work.

**Changed**

- `GitHubClient.merge` still refuses, now because it has no authorization
  attached rather than because merging is forbidden. Execution goes through
  `merge_proposal`, which requires the policy that permitted it.
- Rollback outranks savings. A move that breaches the service level reverts
  regardless of what it saved, because a placement that misses its bound is
  not a cheap placement.
- The kill switch is deleting `.berth/classes.yaml`. An autonomy system whose
  off switch requires contacting the vendor is not one anyone should install.

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
