# Placement status

Generated 2026-08-10 19:00 UTC by berth pilot in shadow mode, nothing is being proposed.

This file is written by the agent and overwritten on every pass. Delete it to stop, or remove the class from `.berth/classes.yaml` to stop watching one.

## Classes

| class | running on | holding | recommended now | basis | proposal |
| --- | --- | --- | --- | --- | --- |
| `corpus-llama3-8b-conversation` | h100-pcie | p99_ttft_ms < 1000 ms | **mi300x**, 49% better | 4 measured, 3 prior |  |
| `corpus-llama3-8b-voice` | l40s | p99_ttft_ms < 300 ms | **mi300x**, 45% better | 4 measured, 3 prior |  |
| `corpus-llama3-8b-longcontext` | mi300x | p99_ttft_ms < 5000 ms | no change | 4 measured, 2 prior |  |
| `corpus-qwen3-moe` | a100-80g | p99_ttft_ms < 2000 ms | not evaluated | | |

## Proposal history

| opened | class | change | trigger | outcome |
| --- | --- | --- | --- | --- |
| 2026-08-10 | `corpus-llama3-8b-conversation` | h100-pcie to mi300x | corpus_cell | shadow |
| 2026-08-10 | `corpus-llama3-8b-voice` | l40s to mi300x | corpus_cell | shadow |

## Sources

| source | last polled |
| --- | --- |
| corpus | 2026-08-10 19:00 |

---

Every figure above is reproducible. `berth place --workload-class <name> --model <key> --incumbent <silicon>` regenerates any recommendation, and the traces behind each measured cell are at docs.reckonresearch.com.
