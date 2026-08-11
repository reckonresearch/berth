```
PILOT  2026-08-11
================================================================

Checked 4 workload classes against 1 sources. 4 source(s) unreachable.

  unreachable  model:NousResearch/Meta-Llama-3-8B: https://huggingface.co/api/models/NousResearch/Meta-Llama-3-8B: HTTP E
  unreachable  model:Qwen/Qwen3-30B-A3B: https://huggingface.co/api/models/Qwen/Qwen3-30B-A3B: HTTP Error 403: 
  unreachable  stack:sgl-project/sglang: https://api.github.com/repos/sgl-project/sglang/releases/latest: HTTP 
  unreachable  stack:vllm-project/vllm: https://api.github.com/repos/vllm-project/vllm/releases/latest: HTTP E
  skipped      corpus-qwen3-moe: unknown model 'qwen3-30b-a3b'

  SAVINGS
  --------------------------------------------------------------
  Nothing yet. No proposal has been merged, so no placement has moved and nothing has been saved. A class we checked and left alone saved nothing either, and reporting the gap to the worst placement in the fleet as a saving would be reporting our own existence as value.

  --------------------------------------------------------------
  4 triggers evaluated. 2 cleared the confidence band and the cost of moving.
  Every figure is reproducible: berth place --workload-class <name>.
```
