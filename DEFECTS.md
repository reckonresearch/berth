# Instrument defects

Every measurement failure found in this project, what caused it, how it was
caught, and the test that fails if it returns.

Published because a measurement tool that has never been caught lying has not
been used hard enough, and because the failures are more useful to anyone else
measuring inference than the results are.

**Six let bad data through. Three rejected good data.** A checker that guards
only one direction is half a checker.

**None were caught by review.** Every one was found by a physical
impossibility, or by asking the operator what command they ran. Care is not a
defence: the floor-subtraction error below was made six times, including
inside documentation warning against it.

---

## The four mechanisms

| Mechanism | Defects | The general form |
| --- | --- | --- |
| Provenance | 2, 3, 8 | a label asserted rather than captured from the source |
| Units | 5, 6 | a quantity used where a different one of the same shape was meant |
| Assumption | 4, 7 | a checker asserting a property the system does not have |
| Ratio | 1 | a constant term left inside a division |
| Assumption | 9 | a guard that blocked the runs it was written to protect |

---

## 1. A constant mistaken for physics

**Mechanism:** ratio. **Direction:** rejected good data.

First-token latency contains a fixed cost that does not scale with prompt
length: scheduler admission, detokenizer setup, the first replay of a compiled
graph. It measures 54.6 ms on an H100 PCIe, 74.6 ms on an L40S, 93 ms under
SGLang on the same L40S.

The validator inverted first-token latency to recover an implied utilisation
without removing it. Implied utilisation then rose with prompt length, because
the constant was most of the measurement at short prompts and a small share at
long ones. The tool reported that attention accounting was wrong. It was not.

**Recurred six times.** A concurrency checker divided batch ratio by time
ratio and reported effective parallelism of 3.2 on a clean file. A trace
auditor computed prefill throughput from raw first-token latency and reported
1.19 times a card's peak FLOPS, also clean. A documentation script written to
warn about this error contained it, and published 1.45 for a quantity whose
real value is 1.01.

**Rule:** subtract the floor before taking any ratio derived from first-token
latency. Fit it with Theil-Sen, never a three-point quadratic, which overshot
on one file and produced a floor above every measurement it was meant to sit
under.

**Pinned by:** `test_defect_1_floor_is_removed_before_inverting_ttft`

---

## 2. Measuring a model nobody asked for

**Mechanism:** provenance. **Direction:** would have let bad data through.

The harness pointed at an OpenAI-compatible endpoint and asked for tokens. It
never asked what the endpoint was serving. A server started with the wrong
checkpoint produces clean timings for a model that is not the one being
attributed, and nothing in the output looks wrong because nothing in it is.

**Fix:** one request to `/v1/models` before the sweep, compared against the
declaration, refusing on mismatch.

**Pinned by:** `test_defect_2_served_model_is_verified_not_assumed`

---

## 3. Hardware asserted rather than observed

**Mechanism:** provenance. **Direction:** let bad data through.

`--silicon` was a command-line argument. A paid operator running the tool
unassisted ran it on an RTX PRO 6000 while declaring an H100. Nothing
objected, because an OpenAI-compatible API does not report what silicon is
underneath it.

Every timing was real. Every one would have entered a public corpus attributed
to hardware that never produced it, and no downstream check could have found
it.

**Fix:** two-tier provenance. Where the server is local, read the GPU identity
and refuse on mismatch. Where it is remote, record `self_reported` and say so.
An unrecognised card records `self_reported` rather than being mapped to the
nearest fleet key, because a guess dressed as a capture is worse than an
absence.

**Pinned by:** `test_defect_3_silicon_mismatch_refuses`,
`test_defect_3_unrecognised_card_is_not_guessed`,
`test_defect_3_remote_endpoint_records_self_reported`

---

## 4. Identical prompts against a default-on cache

**Mechanism:** assumption. **Direction:** let bad data through. **Severity:
highest.**

Every request sent the same prompt, byte for byte, across every request in a
batch, every repetition, and every cell of the same length. Current vLLM and
SGLang enable automatic prefix caching by default, so the first request
prefilled and the rest were served from cache.

- Implied prefill throughput reached **twenty-one times** a card's peak FLOPS
- Fifteen times the prompt tokens cost 1.3 times the time
- Thirty-two times the batch cost 2.3 times the time
- Above batch 1, requests shared cache blocks, so the key-value term appeared
  overcounted by up to 1.8 times, and the error grew with batch in a way that
  looked exactly like a modelling gap at high concurrency

The run completed and the numbers looked plausible to anyone not checking them
against the speed of the hardware.

**Fix:** unique random prefix per request, longer than a cache block. Plus a
startup probe that measures whether the deployment has caching enabled and
records it, because a cell measured with caching on does not transfer to one
without it.

**Pinned by:** `test_defect_4_prompts_are_unique_per_request`,
`test_defect_4_prefill_impossibility_is_caught`

---

## 5. A microbenchmark passed as a datasheet peak

**Mechanism:** units. **Direction:** rejected good data.

Effective bandwidth is conventionally a fraction of a card's datasheet peak. A
microbenchmarked device-to-device copy figure was passed in that slot instead.
It understates achievable read bandwidth, so every ratio came out above 1.0,
and the auditor reported a clean L40S file as contaminated on every batch.

**Fix:** the two are now different types. `Ceiling.datasheet` supports
`efficiency_of`; `Ceiling.microbench` supports `headroom_of` and raises if
asked for an efficiency.

**Pinned by:** `test_defect_5_microbench_cannot_be_used_as_a_datasheet_peak`,
and `test_gigabytes_mistaken_for_terabytes_is_refused`, which guards the
adjacent error: passing 2000 where 2.0 was meant. Not made yet, and now
unmakeable.

---

## 6. An efficiency fitted across a device boundary

**Mechanism:** units. **Direction:** rejected good data.

In a leave-one-silicon-out split there are no training rows for the held-out
card. The code fell back to the other card's timings and divided them by the
held-out card's peak, which is a unit error across a device boundary. It
reported 109.5 percent error with fitted efficiencies of 0.356 and 1.761.

An efficiency above 1.0 is the only reason it was caught. At 0.62 it would
have been written up as a finding.

**Fix:** `efficiency_of` refuses a measurement from a different device.
Carrying a constant between cards is a separate, named operation,
`transfer_efficiency`, which fits against the training card's own peak and
applies the held-out card's, and which refuses any value that is not a
fraction of a peak.

**Pinned by:** `test_defect_6_efficiency_cannot_be_fitted_across_a_device_boundary`,
`test_defect_6_transfer_is_the_supported_operation`,
`test_defect_6_impossible_efficiency_is_refused_at_the_boundary`

---

## 7. One bandwidth assumed per device

**Mechanism:** assumption. **Direction:** rejected good data.

The auditor checked that effective bandwidth is constant across context,
because it is a device property. A device can have two: contiguous weight
reads and scattered paged key-value reads. On an MI300X they differ by six
times above batch 8, while both NVIDIA cards measured hold flat.

The checker reported the first cross-vendor result in the corpus as a
corrupted file.

**Fix:** the direction separates the two cases. Cache contamination credits
the key-value term with bytes that were not moved, so apparent efficiency
**rises** with context. A slow gather makes those bytes dearer, so it
**falls**. Rising is still a failure. Falling is a note with the measurement
attached, and the key-value path rate is now reported per batch level whether
or not anything fails.

**Pinned by:** `test_defect_7_two_access_patterns_are_not_a_corrupted_file`

---

## 8. A quantization declared and not served

**Mechanism:** provenance. **Direction:** let bad data through.

An fp8 cell recorded `kv_bytes=1.0` while the server ran a bf16 key-value
cache. The operator passed `--quantization fp8` and not `--kv-cache-dtype
fp8`. Every timing was real and the label was wrong, which is defect 3 in a
different field.

Caught by asking the operator what command he ran, after the timings showed a
1.2x improvement where fp8 weights should have given closer to 2x.

**Fix:** the run script captures `--quantization`, `--kv-cache-dtype` and
`--dtype` from the launch command, prints them beside the declared byte
counts, and exits non-zero when they disagree. The auditor checks the same
thing from the other direction: at batch 1 with short context, time per output
token is set by weight bytes, so a run declaring half precision that is not
materially faster did not quantize.

**Pinned by:** `test_defect_8_quantization_label_must_match_the_timing`

---

## 9. A guard that blocked the runs it protected

**Mechanism:** assumption. **Direction:** rejected good data.

The quantization guard added for defect 8 read the server's launch flags with
`ps aux | grep`. The script runs under `set -euo pipefail`, and grep exits 1
when it matches nothing, so the assignment failed and the script exited
silently before measuring anything.

A bf16 server has neither quantization flag. The check written to catch
mislabelled runs therefore made the ordinary case the broken one, and two of
three commissioned cells did not run. The only configuration that worked was
the one with both flags present.

Found by the operator, who noticed the pattern across three attempts and
reported it precisely: it only worked when both flags were on the command
line.

**Fix:** `|| true` on both captures. Trivial, and the reason it was missed is
worth more than the fix: the guard had a test for the case it was meant to
refuse and no test for the case it was meant to allow. Every one of the four
combinations is now pinned.

**Rule:** a guard needs a test for what it permits, not only for what it
blocks. Three of nine defects here rejected correct data, and each of them
had a test for the failure path only.

**Pinned by:** `test_defect_9_bf16_run_is_not_blocked_by_the_quantization_guard`,
`test_defect_9_fp8_weights_only_is_not_blocked`,
`test_defect_9_guard_still_refuses_a_label_the_server_contradicts`,
`test_defect_9_fully_declared_fp8_passes`

---

## What actually prevents the next one

**Physical impossibility, not statistical thresholds.** A card cannot exceed
its own peak FLOPS. An efficiency cannot exceed 1.0. First-token latency
cannot be flat in prompt length. These are statements about hardware, and a
file violating one is wrong regardless of how plausible it looks.

**Capture rather than assert.** Any field that can be declared must have a
capture function or be explicitly recorded as unverifiable. Three of the eight
defects were the same field problem in three different fields.

**Types where two quantities share a shape.** Both unit defects were floats
handed to a function expecting a different float. They are now impossible at
the call site rather than detectable afterwards.

**Guard both directions, and test both.** Three of the nine rejected correct
measurements, and every one had a test for the path it was meant to block and
none for the path it was meant to allow. A
checker that only catches bad data will eventually discard a finding, and the
one it discarded here was the most valuable cell in the corpus.

**Ask the operator what command they ran.** Defect 8 was found that way and
nothing else would have found it.

**And a regression test per defect.** A fix that is not pinned is a defect
waiting for a refactor. The register is `tests/test_defect_register.py`, and
adding to it is part of fixing anything found from here.
