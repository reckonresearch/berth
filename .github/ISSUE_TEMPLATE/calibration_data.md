---
name: Calibration data submission
about: Contribute measured traces for a silicon/provider cell
---
**Silicon / provider / region**

**Server + version** (e.g. vLLM x.y.z), **model id**, **device clocks** (nvidia-smi -q excerpt)

**Attach:** traces JSONL (schema v1) + `bench.validate` output + `bench.microbench` ceilings.
Submissions failing physics gates are rejected automatically; see CONTRIBUTING.md rule 4.
