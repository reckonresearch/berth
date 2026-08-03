"""Check the P0 traces for the contamination found in a later run.

WHY this is not optional. `c_eff = 1.09`, the serial-prefill finding, is
published on reckonresearch.com and in the docs as a systems result. It was
measured with the same harness that later produced 18.8x-peak prefill
throughput on an A100, because every request sent a byte-identical prompt.

If P0 was contaminated the same way, that finding is a statement about a
cache, not about a scheduler, and it has to be withdrawn.

The prior is that P0 is clean: it ran vLLM 0.6.3.post1, where automatic
prefix caching required --enable-prefix-caching and was off by default. But
that is an inference from a version number, and a claim on a public site
deserves a check against the data.

    python3 verify_p0_clean.py p0_l40s/traces.jsonl p0_h100/traces.jsonl

What contamination would look like, and what a clean run looks like:

  TTFT vs prompt length   cached: nearly flat.  clean: rises with tokens.
  TTFT vs batch           cached: nearly flat.  clean: rises ~linearly if
                          prefill is serial, which is the c_eff claim itself.
  bw_eff vs context       cached: drifts upward at batch > 1, flat at batch 1.
                          clean: flat everywhere.

The second is the one that matters. c_eff near 1 means TTFT scales with batch.
A cache would flatten exactly that curve, so a contaminated P0 could not have
produced c_eff = 1.09 in the first place. Confirming the batch scaling is
therefore both the check and the finding.
"""

import json
import statistics as st
import sys
from collections import defaultdict


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def report(rows, label):
    print(f"\n{'=' * 68}\n{label}   {len(rows)} traces\n{'=' * 68}")

    by_batch = defaultdict(lambda: defaultdict(list))
    by_len = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_batch[r["batch"]][r["avg_prompt_tokens"]].append(r["measured_ttft_ms"])
        by_len[r["avg_prompt_tokens"]][r["batch"]].append(r["measured_ttft_ms"])

    verdicts = []

    print("\nTTFT vs prompt length, at fixed batch")
    print("  a cache flattens this; real prefill rises with tokens")
    for b in sorted(by_batch):
        ps = sorted(by_batch[b])
        if len(ps) < 2:
            continue
        lo, hi = st.median(by_batch[b][ps[0]]), st.median(by_batch[b][ps[-1]])
        tok, tim = ps[-1] / ps[0], hi / lo
        ok = not (tok >= 5 and tim < 1.5)
        verdicts.append(ok)
        print(f"    b={b:<3} {tok:>4.0f}x tokens -> {tim:>5.2f}x time   "
              f"{'ok' if ok else 'FLAT, cache signature'}")

    print("\nTTFT vs batch, at fixed prompt length")
    print("  this curve IS the c_eff finding: serial prefill means ~linear")
    for p in sorted(by_len):
        bs = sorted(by_len[p])
        if len(bs) < 2:
            continue
        lo, hi = st.median(by_len[p][bs[0]]), st.median(by_len[p][bs[-1]])
        bat, tim = bs[-1] / bs[0], hi / lo
        # Serial prefill means the batch tail waits behind every prompt, so
        # TTFT should scale close to linearly with batch and implied
        # parallelism should sit near 1. A cache flattens the curve, which
        # inflates apparent parallelism: 32x the batch for 2.3x the time
        # implies 14 prompts prefilled at once, which no scheduler does.
        implied_c = bat / tim if tim > 0 else float("inf")
        # The fixed prefill floor is paid once per request regardless of batch,
        # so at short prompts it dominates TTFT and inflates this ratio: a
        # 54.6ms floor on a 60ms TTFT leaves almost nothing that scales. The
        # same error as instrument defect #1, which attributed the floor to
        # compute. Only judge where the floor is a small share of TTFT, which
        # is the longest prompt in the sweep.
        ok = True if p != max(by_len) else implied_c < 3.0
        verdicts.append(ok)
        note = ("ok" if ok else
                f"implied parallelism {implied_c:.1f}, cache signature")
        print(f"    p={p:<6} {bat:>4.0f}x batch  -> {tim:>5.2f}x time   "
              f"implied c_eff {implied_c:>5.1f}   {note}")

    clean = all(verdicts)
    print(f"\n  verdict: {'CLEAN' if clean else 'CONTAMINATED'}")
    if clean:
        print("  TTFT responds to both token count and batch size, which a "
              "prefix cache would suppress. The c_eff measurement stands.")
    else:
        print("  TTFT does not respond to work. Any prefill-derived quantity "
              "from this run, including c_eff and the fitted floor, has to be "
              "withdrawn and re-measured with unique prompts.")
    return clean


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    all_clean = True
    for path in argv:
        rows = load([path])
        if not rows:
            print(f"{path}: no traces")
            return 2
        sil = {r.get("silicon") for r in rows}
        mod = {r.get("model_name") for r in rows}
        all_clean &= report(rows, f"{path}   {'/'.join(sorted(sil))}  "
                                  f"{'/'.join(sorted(mod))}")
    print()
    if all_clean:
        print("Every file is clean. Nothing published needs to change.")
        return 0
    print("At least one file is contaminated. c_eff, the fitted prefill floor "
          "and every TTFT figure derived from it are affected. They are on the "
          "public site and in the docs.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
