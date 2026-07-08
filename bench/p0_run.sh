#!/usr/bin/env bash
# P0 orchestrator: run once per rented box. Everything the runbook does,
# one command, fail-fast, results packaged for bench/data/.
#
# Usage:
#   ./bench/p0_run.sh <silicon> <model> <model_id> [base_url]
#   ./bench/p0_run.sh h100-sxm llama3-8b meta-llama/Meta-Llama-3-8B http://localhost:8000
#
# PREREQUISITE: run ./bench/setup_env.sh ONCE on a fresh pod BEFORE starting
# the server (fixes pyairports/HF-cache/Xet traps). Then start vLLM:
# Assumes a vLLM OpenAI server is already running (pin the version!):
#   python -m vllm.entrypoints.openai.api_server --model <model_id> --max-num-seqs 64
set -euo pipefail

SILICON="${1:?silicon (e.g. h100-sxm)}"
MODEL="${2:?model preset (e.g. llama3-8b)}"
MODEL_ID="${3:?served model id}"
BASE_URL="${4:-http://localhost:8000}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="p0_${SILICON}_${MODEL}_${STAMP}"
mkdir -p "$OUT"
echo "== P0 run -> $OUT =="

# --- ensure pyairports import works (vLLM 0.6.x guided-decoding pulls a broken
# --- pyairports 0.0.1 that pip marks "satisfied" but ships no importable module).
python - <<'PYSTUB'
import importlib.util, os, sys
if importlib.util.find_spec("pyairports") is None or True:
    import site
    sp = site.getsitepackages()[0] if hasattr(site,"getsitepackages") else "/usr/local/lib/python3.11/dist-packages"
    d = os.path.join(sp, "pyairports")
    try:
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d,"__init__.py"),"w").close()
        with open(os.path.join(d,"airports.py"),"w") as f:
            f.write("AIRPORT_LIST = []\n")
    except Exception as e:
        print("pyairports stub warn:", e)
PYSTUB

# 1. Server health — fail before spending an hour measuring a dead endpoint.
curl -sf "${BASE_URL}/v1/models" > "$OUT/server_models.json" \
  || { echo "FATAL: no server at ${BASE_URL}"; exit 1; }

# 2. Provenance metadata — the software stack IS what we're measuring.
{
  echo "date_utc: $STAMP"; echo "silicon: $SILICON"; echo "model: $MODEL"
  echo "model_id: $MODEL_ID"; echo "base_url: $BASE_URL"
  echo "quant: w_bytes=${WEIGHT_BYTES:-2.0} kv_bytes=${KV_BYTES:-2.0}"
  echo "--- vllm ---"; pip show vllm 2>/dev/null | head -2 || echo "vllm: not local"
  echo "--- gpu ---"
  nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,power.limit \
    --format=csv 2>/dev/null || rocm-smi 2>/dev/null || echo "no gpu query tool"
} > "$OUT/meta.txt"

# 3. Ceiling probes (needs torch on the box; warn-don't-die if absent).
if python -c "import torch" 2>/dev/null; then
  python -m bench.microbench | tee "$OUT/microbench.txt"
  BW=$(awk '/bandwidth ceiling/ {print $3}' "$OUT/microbench.txt" || true)
  FLOPS=$(awk '/GEMM ceiling/ {print $3}' "$OUT/microbench.txt" || true)
else
  echo "WARN: torch unavailable — skipping ceilings (validator runs without the ceiling gate)"
  BW=""; FLOPS=""
fi

# 4. Sweep (randomized order, warm-up discarded, usage-based token counts).
# Quant of the served model, bytes per weight/KV element: WEIGHT_BYTES/KV_BYTES
# (bf16=2, fp8/int8=1, fp4/int4=0.5). Recorded per trace so a fp8 cell is
# inverted as fp8, never silently compared to bf16.  Example (fp8 weights):
#   WEIGHT_BYTES=1.0 ./bench/p0_run.sh mi300x qwen3-235b-moe Qwen/Qwen3-235B-A22B-FP8
python -m bench.run_sweep --base-url "$BASE_URL" --silicon "$SILICON" \
  --model "$MODEL" --model-id "$MODEL_ID" --out "$OUT/traces.jsonl" \
  --weight-bytes "${WEIGHT_BYTES:-2.0}" --kv-bytes "${KV_BYTES:-2.0}"

# 5. Term-by-term physics validation (with ceilings when available).
if [[ -n "$BW" && -n "$FLOPS" ]]; then
  python -m bench.validate "$OUT/traces.jsonl" \
    --bw-ceiling-gbps "$BW" --flops-ceiling-tflops "$FLOPS" \
    | tee "$OUT/validate.txt"
else
  python -m bench.validate "$OUT/traces.jsonl" | tee "$OUT/validate.txt"
fi

# 6. Package for bench/data/ in the repo.
tar czf "${OUT}.tar.gz" "$OUT"
echo "== DONE: ${OUT}.tar.gz — commit contents to bench/data/ =="
