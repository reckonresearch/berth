# Running the harness on real hardware (the P0 runbook)

One box, ONE command once the server is up:
`./bench/p0_run.sh <silicon> <model> <model_id> [base_url]` — health check,
provenance capture (versions, clocks), microbench ceilings, randomized
sweep, physics validation with ceiling gates, tarball for bench/data/.
Then across all boxes: `python -m bench.report *.jsonl --out REPORT.md`
generates the publishable report (calibration + CIs + premium tables).
The manual steps, for understanding what the orchestrator does: Stage on a cheap GPU (e.g. L40S ~$1/hr) before
running the full matrix.

1. Boot a vLLM OpenAI-compatible server. Pin and record the version — the
   software stack is part of what you are measuring:
   `python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B --max-num-seqs 64`
2. Ceiling probes (feed results to the validator):
   `python -m bench.microbench`
3. Sweep, then validate:
   `python -m bench.run_sweep --base-url http://localhost:8000 --silicon h100-sxm --model llama3-8b --model-id meta-llama/Meta-Llama-3-8B --out traces.jsonl`
   `python -m bench.validate traces.jsonl --bw-ceiling-gbps <from step 2> --flops-ceiling-tflops <from step 2>`

Measurement-validity rules: run the client on-box (no network jitter in
TTFT); set `--max-num-seqs >= largest batch` (server queueing contaminates
prefill fits); reach thermal steady state; record `nvidia-smi -q` clocks;
sweep order is randomized by the harness (do not "fix" it back).

The validator is term-by-term (bandwidth, KV slope, prefill, residual
structure) with identifiability gates — an underpowered cell reports SKIP,
not a verdict. `calibrate()` on the resulting JSONL updates fleet priors
with 95% CIs.
