"""Benchmark harness: turns rented GPU-hours into TraceRecord JSONL.

Drives an OpenAI-compatible completion server (vLLM/SGLang) with streaming
requests, measures TTFT (time to first token) and TPOT (median inter-token
gap) per concurrency/context cell, and writes traces that `berth.calibrate`
consumes directly. Stdlib-only (http.client + threads) so the box you rent
needs nothing beyond Python.

Mock mode (--mock) generates measurements from the analytical model + noise:
the full pipeline (sweep -> JSONL -> calibrate -> MAPE) is testable with zero
GPUs, and doubles as the CI smoke test. On real hardware the ONLY thing that
changes is where the numbers come from.

Usage (real):
    python -m bench.run_sweep --base-url http://HOST:8000 --silicon h100-sxm \
        --model llama3-8b --model-id meta-llama/Meta-Llama-3-8B --out traces.jsonl
Usage (mock):
    python -m bench.run_sweep --mock --silicon h100-sxm --model llama3-8b --out traces.jsonl
"""

import argparse
import http.client
import json
import math
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from berth.silicon import FLEET
from berth.traces import TraceRecord
from berth.workload import MODELS, WorkloadSpec, profile

DEFAULT_GRID = {
    "batch": [1, 4, 8, 16, 32],
    "prompt": [512, 2048, 7680],
    "output": [128, 256],
    "reps": 3,
}


# ---------------------------------------------------------------- real path

def _stream_one(base_url: str, model_id: str, prompt_tokens: int, max_tokens: int):
    """One streaming completion; returns (ttft_s, tpot_s median). Fails loud."""
    u = urlparse(base_url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=600)
    body = json.dumps({
        "model": model_id,
        # 'x ' pairs tokenize ~1 token each on Llama tokenizers; close enough
        # for load shaping — real prompt length is whatever the server reports.
        "prompt": "x " * prompt_tokens,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},   # server-reported token counts
        "ignore_eos": True,          # vLLM: force exactly max_tokens of decode
    })
    t0 = time.perf_counter()
    conn.request("POST", "/v1/completions", body,
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"server returned {resp.status}: {resp.read()[:200]!r}")
    # SSE framing note: http.client yields arbitrary byte *blocks*, not lines,
    # and events are \n\n-delimited. We buffer bytes and emit only complete
    # events. (Prior versions iterated `for raw in resp` and split on single
    # \n, which fragmented events and silently dropped every token.)
    ttft = None
    first_token_t = None
    last_token_t = None
    n_tokens = 0
    usage = None
    buf = b""
    done = False
    while not done:
        block = resp.read(1024)
        if not block:
            break
        buf += block
        while b"\n\n" in buf:
            event, buf = buf.split(b"\n\n", 1)
            for line in event.split(b"\n"):
                if not line.startswith(b"data: "):
                    continue
                payload = line[len(b"data: "):]
                if payload == b"[DONE]":
                    done = True
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                # `include_usage` attaches usage to EVERY chunk here, including
                # text-bearing ones. Capture it, but decide token-vs-skip on
                # whether a text delta is present (not on usage's presence).
                if chunk.get("usage"):
                    usage = chunk["usage"]
                ch = chunk.get("choices") or []
                if not ch or not ch[0].get("text"):
                    continue
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                    first_token_t = now
                last_token_t = now
                n_tokens += 1
            if done:
                break
    conn.close()
    if ttft is None or n_tokens < 2:
        raise RuntimeError("stream produced <2 tokens; check model_id / server logs")
    # TPOT = decode wall-clock span / inter-token intervals. Timing the DECODE
    # PHASE and dividing by real token count is insensitive to how the server
    # batches SSE flushes. (Prior versions took median of inter-CHUNK gaps,
    # which measured buffer-flush latency ~20us, not the ~15ms/token decode
    # cadence — yielding TPOT ~1000x too fast and corrupting every downstream
    # bandwidth/KV/MFU fit.) Prefer the server's completion_tokens when present.
    served_tokens = usage.get("completion_tokens") if usage else None
    n_for_tpot = served_tokens if served_tokens and served_tokens >= 2 else n_tokens
    tpot = (last_token_t - first_token_t) / max(1, n_for_tpot - 1)
    prompt_tokens = usage.get("prompt_tokens") if usage else None
    return ttft, tpot, prompt_tokens


def measure_cell(base_url, model_id, batch, prompt, output):
    """Fire `batch` concurrent streams; return (ttft_ms, tpot_ms, prompt_toks)."""
    with ThreadPoolExecutor(max_workers=batch) as ex:
        results = list(ex.map(
            lambda _: _stream_one(base_url, model_id, prompt, output),
            range(batch),
        ))
    ttfts = [r[0] for r in results]
    tpots = [r[1] for r in results]
    reported = [r[2] for r in results if r[2] is not None]
    prompt_toks = round(statistics.median(reported)) if reported else prompt
    return statistics.median(ttfts) * 1e3, statistics.median(tpots) * 1e3, prompt_toks


# ---------------------------------------------------------------- mock path

def measure_cell_mock(silicon, model_name, batch, prompt, output, rng):
    """Analytical truth + 5% lognormal noise: validates the pipeline, not the model."""
    from berth.estimate import estimate
    hw = FLEET[silicon]
    sig = profile(WorkloadSpec(model=MODELS[model_name], target_batch=batch,
                               avg_prompt_tokens=prompt, avg_output_tokens=output))
    e = estimate(sig, hw, hw.base_price_hr)
    if not e.feasible:
        return None
    n = lambda: math.exp(rng.gauss(0.0, 0.05))
    return e.ttft_ms * n(), e.tpot_ms * n()


# ------------------------------------------------------------------- sweep

def run_sweep(args) -> list[TraceRecord]:
    rng = random.Random(args.seed)
    traces: list[TraceRecord] = []
    total = 0
    # Randomize cell order: fixed grid order confounds thermal/clock drift
    # with batch/context position. Shuffle is seeded -> reproducible.
    cells = [(b, p, o, r)
             for b in args.grid["batch"]
             for p in args.grid["prompt"]
             for o in args.grid["output"]
             for r in range(args.grid["reps"])]
    rng.shuffle(cells)
    warmed: set[tuple] = set()   # cell keys whose cold-cache run was discarded
    for batch, prompt, output, _rep in cells:
                    if args.mock:
                        m = measure_cell_mock(args.silicon, args.model,
                                              batch, prompt, output, rng)
                        if m is None:
                            continue
                        ttft_ms, tpot_ms = m
                    else:
                        # Warm-up rep is measured but discarded (cold caches).
                        ttft_ms, tpot_ms, prompt_actual = measure_cell(
                            args.base_url, args.model_id, batch, prompt, output)
                        prompt = prompt_actual   # record truth, not the request
                        # Discard the first chronological execution of each
                        # cell (cold caches, CUDA-graph capture). The shuffled
                        # rep label is NOT chronology — keying on rep==0 after
                        # randomization discards a random rep instead.
                        key = (batch, prompt, output)
                        if key not in warmed and args.grid["reps"] > 1:
                            warmed.add(key)
                            continue
                    traces.append(TraceRecord(
                        silicon=args.silicon, model_name=args.model,
                        batch=batch, avg_prompt_tokens=prompt,
                        avg_output_tokens=output,
                        measured_ttft_ms=ttft_ms, measured_tpot_ms=tpot_ms,
                        t=total / 1000.0,
                    ))
                    total += 1
                    print(f"[{total}] b={batch} p={prompt} o={output} "
                          f"TTFT={ttft_ms:.0f}ms TPOT={tpot_ms:.1f}ms")
    return traces


SCHEMA_VERSION = 1  # bump on any TraceRecord field change; loaders must check


def save_jsonl(traces: list[TraceRecord], path: str) -> None:
    with open(path, "w") as f:
        for tr in traces:
            f.write(json.dumps({"schema": SCHEMA_VERSION, **tr.__dict__}) + "\n")


def load_jsonl(path: str) -> list[TraceRecord]:
    """Versioned loader. Unversioned lines are treated as v1 (pre-versioning
    files from this repo only); unknown versions fail loud — silently
    misparsing external contributions is how an index corrupts."""
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            v = d.pop("schema", 1)
            if v != SCHEMA_VERSION:
                raise ValueError(f"trace schema v{v} unsupported (loader is v{SCHEMA_VERSION})")
            out.append(TraceRecord(**d))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--silicon", required=True, choices=sorted(FLEET))
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--base-url", help="OpenAI-compatible server (real mode)")
    p.add_argument("--model-id", help="served model id, e.g. meta-llama/... (real mode)")
    p.add_argument("--mock", action="store_true", help="analytical truth + noise, no GPU")
    p.add_argument("--out", default="traces.jsonl")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if not args.mock and not (args.base_url and args.model_id):
        p.error("real mode requires --base-url and --model-id (or pass --mock)")
    args.grid = DEFAULT_GRID
    traces = run_sweep(args)
    save_jsonl(traces, args.out)
    print(f"wrote {len(traces)} traces -> {args.out}")


if __name__ == "__main__":
    main()
